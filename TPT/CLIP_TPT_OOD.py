
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision.transforms import functional
from PIL import Image
from copy import deepcopy
import torch.backends.cudnn as cudnn
from transformers import CLIPModel, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPTokenizer
import os
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import torch
import random
from modeling_attn_mask_utils import _create_4d_causal_attention_mask   #这是拿来干什么的
import torch
import torch.nn as nn
from datasets import load_dataset
from collections import defaultdict
from datasets import Dataset
from tqdm import tqdm
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dataset_loading(dataset_name, num_samples_per_class=1, cache_dir='./dataset/', trust_remote_code=True):
    if os.path.exists(cache_dir):
        print("Loading subset from disk...")
    else:
        print("Downloading and processing dataset...")

        # Load the dataset with streaming
        dataset = load_dataset(dataset_name, split='train',
                               streaming=True, cache_dir=cache_dir, trust_remote_code=True)

        # Initialize a dictionary to hold samples for each class
        samples = defaultdict(list)

        # Class label column name
        class_label_column = 'label'  # Adjust according to your dataset

        # Initialize tqdm progress bar
        progress_bar = tqdm(dataset, desc="Downloading data", unit=" example")

        # Iterate over the dataset and collect samples
        # Iterate over the dataset and collect samples
        for example in progress_bar:
            label = example[class_label_column]

            if len(samples[label]) < num_samples_per_class:
                samples[label].append(example)

            # Stop if we have enough samples for all classes
            if all(len(samples[label]) >= num_samples_per_class for label in samples):
                continue

        # Convert the samples dictionary to a list of examples
        sampled_examples = [example for examples in samples.values()
                            for example in examples]

        # Convert the list of sampled examples to a Hugging Face dataset
        sampled_dataset = Dataset.from_list(sampled_examples)

        # Save the dataset to disk
        sampled_dataset.save_to_disk(cache_dir)
        print("Dataset saved to disk")

    return

def predict(model, images, prompts):
    inputs = processor(text=prompts, images=images,
                       return_tensors="pt", padding=True).to(device)

    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    return logits_per_image


def get_clip_results(dataset, model, prompts):
    confidence = []
    correct = []

    num_correct = 0
    total_samples = 0

    # Prepare tqdm progress bar
    progress_bar = tqdm(total=len(dataset), desc="Evaluating", unit="image")

    with torch.no_grad():
        for example in dataset:
            image = example['image']
            label = example['label']

            output = predict(model, image, prompts) #输出的logits维度为200

            # Accuracy
            pred = output.argmax(dim=1).item()
            is_correct = (pred == label)

            num_correct += is_correct
            total_samples += 1

            confidence.extend(F.softmax(output, dim=1).max(1)[0].cpu().numpy())
            correct.append(is_correct)

            # Update progress bar
            progress_bar.update(1)

    progress_bar.close()
    return num_correct / total_samples, confidence, correct
class PromptLearner(nn.Module):
    def __init__(self, text_encoder, tokenizer, classnames, context_length, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = text_encoder.get_input_embeddings().weight.dtype
        print(f"dtype is {dtype}")
        self.dtype = dtype
        self.device = text_encoder.device
        ctx_dim = text_encoder.config.hidden_size
        # print(f"text hidden_embedding dimension is {ctx_dim}")
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size
        # print(f"batch_size is {batch_size}")
        self.context_length = context_length
        print(f"context_length is {context_length}")

        if ctx_init:
            # use given words to initialize context vectors
            print(
                "Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" ")) #用空格隔开形成list
            prompt = tokenizer(
                ctx_init, return_tensors='pt', padding='max_length', max_length=self.context_length, truncation=True).input_ids.to(self.device) #前缀的tokenize
            # Decode the token IDs back to text
            # decoded_prompt = tokenizer.decode(
            #     prompt[0], skip_special_tokens=False)
            # print(f"Decoded prompt: {decoded_prompt}")

            with torch.no_grad():
                embedding = text_encoder.get_input_embeddings()(prompt).type(dtype) #这里加torch.no_grad()有什么作用吗
                # print(f"shape of embedding: {embedding.shape}")
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            # print(f"Shape of ctx_vectors: {ctx_vectors.shape}")
            # print(f"Length of initial context state of the first prompt: {
            #       len(ctx_vectors[0])}")
            # print(f"Initial context state of the first prompt: {
            #       ctx_vectors[0]}")
            # print(f"Length of initial context state of the second prompt: {
            #       len(ctx_vectors[1])}")
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        print(f"Number of classes: {n_cls}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None:
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  # (N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()    #init时进行了一次备份，以便后续还原
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized  #前缀

        if not self.learned_cls:
            print("Initializing a class token from list of class names")
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(tokenizer.tokenize(name)) for name in classnames]
            # print(f"Length of the first class names: {name_lens[0]}")
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            # assume each learnable cls_token is only 1 word
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype)
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"   #这个X是什么意思，代表未知类别吗
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " +
                       cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors)  # to be optimized

        tokenized_prompts = torch.cat(
            [tokenizer(p, return_tensors='pt', padding='max_length', max_length=self.context_length, truncation=True).input_ids.to(self.device) for p in prompts])#完整prompts的tokenize
        with torch.no_grad():
            embedding = text_encoder.get_input_embeddings()(tokenized_prompts).type(dtype)
            # print(f"Length of embedding: {len(embedding)}")
            # print(f"Length of the first embedding: {len(embedding[0])}")

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer(
                "token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer(
                "token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors)  # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, tokenizer, text_encoder):  #选中所需的类
        self.n_cls = len(classnames)
        print(f"New number of classes: {self.n_cls}")
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(tokenizer.tokenize(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " +
                       name + "." for name in classnames]
        else:
            # assume each learnable cls_token is only 1 word
            cls_vectors = torch.empty(
                self.n_cls, 1, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(cls_vectors, std=0.02)   #这里的cls_vectors好像随机初始化了多次，是正确的吗，在__init__中初始化了一次在reset_classnames中又初始化了一次
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " +
                       cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()

        tokenized_prompts = torch.cat(
            [tokenizer(p, return_tensors='pt', padding='max_length', max_length=self.context_length, truncation=True).input_ids.to(self.device) for p in prompts])

        with torch.no_grad():
            embedding = text_encoder.get_input_embeddings()(
                tokenized_prompts).type(self.dtype)
            # print(f"New shape of embedding after reset: {embedding.shape}")

        self.token_prefix = embedding[:, :1, :]
        # print(f"New Shape of token_prefix: {self.token_prefix.shape}")
        self.token_suffix = embedding[:, 1 + self.n_ctx:, :]  # CLS, EOS
        # print(f"New Shape of token_suffix: {self.token_suffix.shape}")

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx
        # print(f"self.n_cls is {self.n_cls}")
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)   #这两行是干嘛的
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1) #这两行是干嘛的

        # print(f"Shape of ctx: {ctx.shape}")

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None:
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        cls,     # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                # split the ctx at the position of [CLS] in `ctx_init`
                half_n_ctx = self.split_idx
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                print(f"name_len is {name_len}")
                prefix_i = prefix[i: i + 1, :, :]
                print(f"prefix_i is {prefix_i}")
                class_i = suffix[i: i + 1, :name_len, :]
                print(f"class_i is {class_i}")
                suffix_i = suffix[i: i + 1, name_len:, :]
                print(f"suffix_i is {suffix_i}")
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                print(f"ctx_i_half1 is {ctx_i_half1}")
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                print(f"ctx_i_half2 is {ctx_i_half2}")
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        # print(f"Shape of prompts: {prompts.shape}")

        return prompts
    
class TestTimePromptTuning(nn.Module):
    def __init__(self, device, image_encoder, text_encoder, tokenizer, logit_scale, context_length, classnames, batch_size,
                 n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super(TestTimePromptTuning, self).__init__()
        self.image_encoder = image_encoder.to(device)
        self.text_encoder = text_encoder.to(device)
        self.tokenizer = tokenizer
        self.logit_scale = logit_scale.to(device)
        self.device = device
        # prompt tuning
        self.prompt_learner = PromptLearner(
            text_encoder, tokenizer, classnames, context_length, batch_size, n_ctx, ctx_init, ctx_position, learned_cls).to(device)

        self.eos_token_id = self.text_encoder.text_model.eos_token_id

    @property
    def dtype(self):
        return next(self.image_encoder.parameters()).dtype

    # Restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, tokenizer, text_encoder):
        self.prompt_learner.reset_classnames(
            classnames, tokenizer, text_encoder)

    def get_text_features(self):
        #这一段好好看看
        # Get the learned prompts from the PromptLearner
        prompts = self.prompt_learner()  #第一次编码
        # print(f"prompts are f{prompts}")
        tokenized_prompts = self.prompt_learner.tokenized_prompts

        input_shape = tokenized_prompts.size()
        input_ids = tokenized_prompts.view(-1, input_shape[-1])  
        # print(f"input_ids are {input_ids}")
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324

        text_embeddings = self.text_encoder.text_model.embeddings(input_ids=input_ids,
                                                                  inputs_embeds=prompts)  #第二次编码

        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, text_embeddings.dtype, device=text_embeddings.device
        )

        # Access the encoder directly
        encoder = self.text_encoder.text_model.encoder
        encoder_outputs = encoder(
            inputs_embeds=text_embeddings, causal_attention_mask=causal_attention_mask) #第三次编码

        final_layer_norm = self.text_encoder.text_model.final_layer_norm

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            pooled_output = last_hidden_state[
                torch.arange(
                    last_hidden_state.shape[0], device=last_hidden_state.device),
                tokenized_prompts.to(
                    dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            pooled_output = last_hidden_state[
                torch.arange(
                    last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer) 用最后一个有效token作为hidden states
                (tokenized_prompts.to(dtype=torch.int,
                 device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]

        # pooled_output_indices = tokenized_prompts.argmax(dim=-1)
        # pooled_output = last_hidden_state[torch.arange(
        #     last_hidden_state.shape[0]), pooled_output_indices]

        text_embeds = self.text_encoder.text_projection(pooled_output)

        return text_embeds

    def inference(self, image):

        with torch.no_grad():
            outputs = self.image_encoder(pixel_values=image.type(self.dtype))
            image_features = outputs.image_embeds.to(self.device)

        text_features = self.get_text_features()

        # normalized features
        image_features = image_features / \
            image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / \
            text_features.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()
        logits_per_text = torch.matmul(text_features, image_features.t().to(text_features.device)) * logit_scale.to(
            text_features.device
        )
        logits_per_image = logits_per_text.t()

        return logits_per_image

    def forward(self, input):
        return self.inference(input)

def augmix(image, preprocess):  #这里涉及的数据增强使用的是随机裁剪，对标groundingdino的region_proposal
    preaugment = transforms.Compose([
        transforms.RandomResizedCrop(336),
        transforms.RandomHorizontalFlip(),
    ])
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)  #后续需要定位该preprocess在哪
    return x_processed


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2): #后续定位n_views，base_transform
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views

    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess) for _ in range(self.n_views)]
        return [image] + views #这里为什么要叠加

def transform_function(examples):
    transformed_images = []
    tpt_batch_size = 64
    resolution = 336
    try:
        from torchvision.transforms import InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC
    except ImportError:
        BICUBIC = Image.BICUBIC

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=BICUBIC),
        transforms.CenterCrop(resolution)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    data_transform = AugMixAugmenter(
        base_transform, preprocess, n_views=tpt_batch_size-1)
    for img in examples['image']:
        img = img.convert("RGB")
        transformed_image = data_transform(img)
        transformed_images.append(transformed_image)

    # Ensure other columns are included in the returned dictionary
    return {'image': transformed_images, **{key: examples[key] for key in examples if key != 'image'}}

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[
        :int(batch_entropy.size()[0] * top)]
    return logits[idx], idx


def avg_entropy(outputs):
    # logits = outputs.log_softmax(dim=1) [N, 1000]
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    # avg_logits = logits.mean(0) [1, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


# tta_steps :test-time-adapt steps
# selection_p : confidence selection percentile
def test_time_tuning(model, inputs, optimizer, scaler, tta_steps=1, selection_p=0.1):
    for j in range(tta_steps):

        with torch.cuda.amp.autocast():

            output = model(inputs)

            # Debug: Print the output to ensure it does not contain NaNs/Infs
            if torch.isnan(output).any() or torch.isinf(output).any():
                raise ValueError("Model output contains NaN or Inf.")

            output, selected_idx = select_confident_samples(
                output, selection_p)

            loss = avg_entropy(output)

        # Ensure the loss is not NaN or Inf
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(
                "Loss value is NaN or Inf, cannot proceed with optimization.")

        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()

        # Print gradients for debugging
        nan_in_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    print(f"No gradient computed for {name}")
                else:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        nan_in_grad = True
                        print(f"NaN or Inf found in gradients for {name}")
                    # else:
                        # print(f"Gradient for {name}: {
                        #       param.grad.norm().item()}")

        if nan_in_grad:
            raise ValueError(
                "NaN or Inf found in gradients, cannot proceed with optimization.")
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()
    return


def predict_with_tpt(model, images, optimizer, scaler, optim_state, tta_steps=1, selection_p=0.1):
    # reset the tunable prompt to its initial state
    with torch.no_grad():
        model.reset()

    optimizer.load_state_dict(optim_state)
    test_time_tuning(model, images, optimizer, scaler)

    image = images[0].unsqueeze(0)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            output = model(image)

    return output


def test_time_adapt_eval(data_loader, model, optimizer, optim_state, scaler, device):
    model.to(device)
    # reset model and switch to evaluate mode
    model.eval()
    with torch.no_grad():
        model.reset()

    confidence = []
    correct = []

    num_correct = 0
    total_samples = 0

    # Initialize the progress bar
    pbar = tqdm(data_loader, desc="Evaluating", unit="batch", leave=False)

    for i, (images, target) in enumerate(pbar):
        images = images.to(device, non_blocking=True)

        target = target.to(device, non_blocking=True)

        output = predict_with_tpt(
            model, images, optimizer, scaler, optim_state)

        # Accuracy
        pred = output.argmax(dim=1).item()

        is_correct = (pred == target.item())
        num_correct += is_correct
        total_samples += 1

        confidence.extend(F.softmax(output, dim=1).max(1)[0].cpu().numpy())
        correct.append(is_correct)

    # Close the progress bar
    pbar.close()

    return num_correct / total_samples, confidence, correct
    

            
def collate_fn(batch):
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]  # Assuming there is a label
    # Stack images and labels
    try:
        images = torch.stack([img for sublist in images for img in sublist])
    except TypeError as e:
        print(f"Error stacking images: {e}")
        print(f"Image batch content: {images}")
        raise

    labels = torch.tensor(labels)
    return images, labels



def test_time_adapt_eval(data_loader, model, optimizer, optim_state, scaler, device):
    model.to(device)
    # reset model and switch to evaluate mode
    model.eval()
    with torch.no_grad():
        model.reset()

    confidence = []
    correct = []

    num_correct = 0
    total_samples = 0

    # Initialize the progress bar
    pbar = tqdm(data_loader, desc="Evaluating", unit="batch", leave=False)

    for i, (images, target) in enumerate(pbar):
        images = images.to(device, non_blocking=True)

        target = target.to(device, non_blocking=True)

        output = predict_with_tpt(
            model, images, optimizer, scaler, optim_state)

        # Accuracy
        pred = output.argmax(dim=1).item()

        is_correct = (pred == target.item())
        num_correct += is_correct
        total_samples += 1

        confidence.extend(F.softmax(output, dim=1).max(1)[0].cpu().numpy())
        correct.append(is_correct)

    # Close the progress bar
    pbar.close()

    return num_correct / total_samples, confidence, correct


if __name__=='__main__':
    thousand_k_to_200 = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: 0, 7: -1, 8: -1, 9: -1, 10: -1, 11: 1, 12: -1, 13: 2, 14: -1, 15: 3, 16: -1, 17: 4, 18: -1, 19: -1, 20: -1, 21: -1, 22: 5, 23: 6, 24: -1, 25: -1, 26: -1, 27: 7, 28: -1, 29: -1, 30: 8, 31: -1, 32: -1, 33: -1, 34: -1, 35: -1, 36: -1, 37: 9, 38: -1, 39: 10, 40: -1, 41: -1, 42: 11, 43: -1, 44: -1, 45: -1, 46: -1, 47: 12, 48: -1, 49: -1, 50: 13, 51: -1, 52: -1, 53: -1, 54: -1, 55: -1, 56: -1, 57: 14, 58: -1, 59: -1, 60: -1, 61: -1, 62: -1, 63: -1, 64: -1, 65: -1, 66: -1, 67: -1, 68: -1, 69: -1, 70: 15, 71: 16, 72: -1, 73: -1, 74: -1, 75: -1, 76: 17, 77: -1, 78: -1, 79: 18, 80: -1, 81: -1, 82: -1, 83: -1, 84: -1, 85: -1, 86: -1, 87: -1, 88: -1, 89: 19, 90: 20, 91: -1, 92: -1, 93: -1, 94: 21, 95: -1, 96: 22, 97: 23, 98: -1, 99: 24, 100: -1, 101: -1, 102: -1, 103: -1, 104: -1, 105: 25, 106: -1, 107: 26, 108: 27, 109: -1, 110: 28, 111: -1, 112: -1, 113: 29, 114: -1, 115: -1, 116: -1, 117: -1, 118: -1, 119: -1, 120: -1, 121: -1, 122: -1, 123: -1, 124: 30, 125: 31, 126: -1, 127: -1, 128: -1, 129: -1, 130: 32, 131: -1, 132: 33, 133: -1, 134: -1, 135: -1, 136: -1, 137: -1, 138: -1, 139: -1, 140: -1, 141: -1, 142: -1, 143: 34, 144: 35, 145: -1, 146: -1, 147: -1, 148: -1, 149: -1, 150: 36, 151: 37, 152: -1, 153: -1, 154: -1, 155: -1, 156: -1, 157: -1, 158: -1, 159: -1, 160: -1, 161: -1, 162: -1, 163: -1, 164: -1, 165: -1, 166: -1, 167: -1, 168: -1, 169: -1, 170: -1, 171: -1, 172: -1, 173: -1, 174: -1, 175: -1, 176: -1, 177: -1, 178: -1, 179: -1, 180: -1, 181: -1, 182: -1, 183: -1, 184: -1, 185: -1, 186: -1, 187: -1, 188: -1, 189: -1, 190: -1, 191: -1, 192: -1, 193: -1, 194: -1, 195: -1, 196: -1, 197: -1, 198: -1, 199: -1, 200: -1, 201: -1, 202: -1, 203: -1, 204: -1, 205: -1, 206: -1, 207: 38, 208: -1, 209: -1, 210: -1, 211: -1, 212: -1, 213: -1, 214: -1, 215: -1, 216: -1, 217: -1, 218: -1, 219: -1, 220: -1, 221: -1, 222: -1, 223: -1, 224: -1, 225: -1, 226: -1, 227: -1, 228: -1, 229: -1, 230: -1, 231: -1, 232: -1, 233: -1, 234: 39, 235: 40, 236: -1, 237: -1, 238: -1, 239: -1, 240: -1, 241: -1, 242: -1, 243: -1, 244: -1, 245: -1, 246: -1, 247: -1, 248: -1, 249: -1, 250: -1, 251: -1, 252: -1, 253: -1, 254: 41, 255: -1, 256: -1, 257: -1, 258: -1, 259: -1, 260: -1, 261: -1, 262: -1, 263: -1, 264: -1, 265: -1, 266: -1, 267: -1, 268: -1, 269: -1, 270: -1, 271: -1, 272: -1, 273: -1, 274: -1, 275: -1, 276: -1, 277: 42, 278: -1, 279: -1, 280: -1, 281: -1, 282: -1, 283: 43, 284: -1, 285: -1, 286: -1, 287: 44, 288: -1, 289: -1, 290: -1, 291: 45, 292: -1, 293: -1, 294: -1, 295: 46, 296: -1, 297: -1, 298: 47, 299: -1, 300: -1, 301: 48, 302: -1, 303: -1, 304: -1, 305: -1, 306: 49, 307: 50, 308: 51, 309: 52, 310: 53, 311: 54, 312: -1, 313: 55, 314: 56, 315: 57, 316: -1, 317: 58, 318: -1, 319: 59, 320: -1, 321: -1, 322: -1, 323: 60, 324: 61, 325: -1, 326: 62, 327: 63, 328: -1, 329: -1, 330: 64, 331: -1, 332: -1, 333: -1, 334: 65, 335: 66, 336: 67, 337: -1, 338: -1, 339: -1, 340: -1, 341: -1, 342: -1, 343: -1, 344: -1, 345: -1, 346: -1, 347: 68, 348: -1, 349: -1, 350: -1, 351: -1, 352: -1, 353: -1, 354: -1, 355: -1, 356: -1, 357: -1, 358: -1, 359: -1, 360: -1, 361: 69, 362: -1, 363: 70, 364: -1, 365: -1, 366: -1, 367: -1, 368: -1, 369: -1, 370: -1, 371: -1, 372: 71, 373: -1, 374: -1, 375: -1, 376: -1, 377: -1, 378: 72, 379: -1, 380: -1, 381: -1, 382: -1, 383: -1, 384: -1, 385: -1, 386: 73, 387: -1, 388: -1, 389: -1, 390: -1, 391: -1, 392: -1, 393: -1, 394: -1, 395: -1, 396: -1, 397: 74, 398: -1, 399: -1, 400: 75, 401: 76, 402: 77, 403: -1, 404: 78, 405: -1, 406: -1, 407: 79, 408: -1, 409: -1, 410: -1, 411: 80, 412: -1, 413: -1, 414: -1, 415: -1, 416: 81, 417: 82, 418: -1, 419: -1, 420: 83, 421: -1, 422: -1, 423: -1, 424: -1, 425: 84, 426: -1, 427: -1, 428: 85, 429: -1, 430: 86, 431: -1, 432: -1, 433: -1, 434: -1, 435: -1, 436: -1, 437: 87, 438: 88, 439: -1, 440: -1, 441: -1, 442: -1, 443: -1, 444: -1, 445: 89, 446: -1, 447: -1, 448: -1, 449: -1, 450: -1, 451: -1, 452: -1, 453: -1, 454: -1, 455: -1, 456: 90, 457: 91, 458: -1, 459: -1, 460: -1, 461: 92, 462: 93, 463: -1, 464: -1, 465: -1, 466: -1, 467: -1, 468: -1, 469: -1, 470: 94, 471: -1, 472: 95, 473: -1, 474: -1, 475: -1, 476: -1, 477: -1, 478: -1, 479: -1, 480: -1, 481: -1, 482: -1, 483: 96, 484: -1, 485: -1, 486: 97, 487: -1, 488: 98, 489: -1, 490: -1, 491: -1, 492: 99, 493: -1, 494: -1, 495: -1, 496: 100, 497: -1, 498: -1, 499: -1, 500: -1, 501: -1, 502: -1, 503: -1, 504: -1, 505: -1, 506: -1, 507: -1, 508: -1, 509: -1, 510: -1,
                     511: -1, 512: -1, 513: -1, 514: 101, 515: -1, 516: 102, 517: -1, 518: -1, 519: -1, 520: -1, 521: -1, 522: -1, 523: -1, 524: -1, 525: -1, 526: -1, 527: -1, 528: 103, 529: -1, 530: 104, 531: -1, 532: -1, 533: -1, 534: -1, 535: -1, 536: -1, 537: -1, 538: -1, 539: 105, 540: -1, 541: -1, 542: 106, 543: 107, 544: -1, 545: -1, 546: -1, 547: -1, 548: -1, 549: 108, 550: -1, 551: -1, 552: 109, 553: -1, 554: -1, 555: -1, 556: -1, 557: 110, 558: -1, 559: -1, 560: -1, 561: 111, 562: 112, 563: -1, 564: -1, 565: -1, 566: -1, 567: -1, 568: -1, 569: 113, 570: -1, 571: -1, 572: 114, 573: 115, 574: -1, 575: 116, 576: -1, 577: -1, 578: -1, 579: 117, 580: -1, 581: -1, 582: -1, 583: -1, 584: -1, 585: -1, 586: -1, 587: -1, 588: -1, 589: 118, 590: -1, 591: -1, 592: -1, 593: -1, 594: -1, 595: -1, 596: -1, 597: -1, 598: -1, 599: -1, 600: -1, 601: -1, 602: -1, 603: -1, 604: -1, 605: -1, 606: 119, 607: 120, 608: -1, 609: 121, 610: -1, 611: -1, 612: -1, 613: -1, 614: 122, 615: -1, 616: -1, 617: -1, 618: -1, 619: -1, 620: -1, 621: -1, 622: -1, 623: -1, 624: -1, 625: -1, 626: 123, 627: 124, 628: -1, 629: -1, 630: -1, 631: -1, 632: -1, 633: -1, 634: -1, 635: -1, 636: -1, 637: -1, 638: -1, 639: -1, 640: 125, 641: 126, 642: 127, 643: 128, 644: -1, 645: -1, 646: -1, 647: -1, 648: -1, 649: -1, 650: -1, 651: -1, 652: -1, 653: -1, 654: -1, 655: -1, 656: -1, 657: -1, 658: 129, 659: -1, 660: -1, 661: -1, 662: -1, 663: -1, 664: -1, 665: -1, 666: -1, 667: -1, 668: 130, 669: -1, 670: -1, 671: -1, 672: -1, 673: -1, 674: -1, 675: -1, 676: -1, 677: 131, 678: -1, 679: -1, 680: -1, 681: -1, 682: 132, 683: -1, 684: 133, 685: -1, 686: -1, 687: 134, 688: -1, 689: -1, 690: -1, 691: -1, 692: -1, 693: -1, 694: -1, 695: -1, 696: -1, 697: -1, 698: -1, 699: -1, 700: -1, 701: 135, 702: -1, 703: -1, 704: 136, 705: -1, 706: -1, 707: -1, 708: -1, 709: -1, 710: -1, 711: -1, 712: -1, 713: -1, 714: -1, 715: -1, 716: -1, 717: -1, 718: -1, 719: 137, 720: -1, 721: -1, 722: -1, 723: -1, 724: -1, 725: -1, 726: -1, 727: -1, 728: -1, 729: -1, 730: -1, 731: -1, 732: -1, 733: -1, 734: -1, 735: -1, 736: 138, 737: -1, 738: -1, 739: -1, 740: -1, 741: -1, 742: -1, 743: -1, 744: -1, 745: -1, 746: 139, 747: -1, 748: -1, 749: 140, 750: -1, 751: -1, 752: 141, 753: -1, 754: -1, 755: -1, 756: -1, 757: -1, 758: 142, 759: -1, 760: -1, 761: -1, 762: -1, 763: 143, 764: -1, 765: 144, 766: -1, 767: -1, 768: 145, 769: -1, 770: -1, 771: -1, 772: -1, 773: 146, 774: 147, 775: -1, 776: 148, 777: -1, 778: -1, 779: 149, 780: 150, 781: -1, 782: -1, 783: -1, 784: -1, 785: -1, 786: 151, 787: -1, 788: -1, 789: -1, 790: -1, 791: -1, 792: 152, 793: -1, 794: -1, 795: -1, 796: -1, 797: 153, 798: -1, 799: -1, 800: -1, 801: -1, 802: 154, 803: 155, 804: 156, 805: -1, 806: -1, 807: -1, 808: -1, 809: -1, 810: -1, 811: -1, 812: -1, 813: 157, 814: -1, 815: 158, 816: -1, 817: -1, 818: -1, 819: -1, 820: 159, 821: -1, 822: -1, 823: 160, 824: -1, 825: -1, 826: -1, 827: -1, 828: -1, 829: -1, 830: -1, 831: 161, 832: -1, 833: 162, 834: -1, 835: 163, 836: -1, 837: -1, 838: -1, 839: 164, 840: -1, 841: -1, 842: -1, 843: -1, 844: -1, 845: 165, 846: -1, 847: 166, 848: -1, 849: -1, 850: 167, 851: -1, 852: -1, 853: -1, 854: -1, 855: -1, 856: -1, 857: -1, 858: -1, 859: 168, 860: -1, 861: -1, 862: 169, 863: -1, 864: -1, 865: -1, 866: -1, 867: -1, 868: -1, 869: -1, 870: 170, 871: -1, 872: -1, 873: -1, 874: -1, 875: -1, 876: -1, 877: -1, 878: -1, 879: 171, 880: 172, 881: -1, 882: -1, 883: -1, 884: -1, 885: -1, 886: -1, 887: -1, 888: 173, 889: -1, 890: 174, 891: -1, 892: -1, 893: -1, 894: -1, 895: -1, 896: -1, 897: 175, 898: -1, 899: -1, 900: 176, 901: -1, 902: -1, 903: -1, 904: -1, 905: -1, 906: -1, 907: 177, 908: -1, 909: -1, 910: -1, 911: -1, 912: -1, 913: 178, 914: -1, 915: -1, 916: -1, 917: -1, 918: -1, 919: -1, 920: -1, 921: -1, 922: -1, 923: -1, 924: 179, 925: -1, 926: -1, 927: -1, 928: -1, 929: -1, 930: -1, 931: -1, 932: 180, 933: 181, 934: 182, 935: -1, 936: -1, 937: 183, 938: -1, 939: -1, 940: -1, 941: -1, 942: -1, 943: 184, 944: -1, 945: 185, 946: -1, 947: 186, 948: -1, 949: -1, 950: -1, 951: 187, 952: -1, 953: -1, 954: 188, 955: -1, 956: 189, 957: 190, 958: -1, 959: 191, 960: -1, 961: -1, 962: -1, 963: -1, 964: -1, 965: -1, 966: -1, 967: -1, 968: -1, 969: -1, 970: -1, 971: 192, 972: 193, 973: -1, 974: -1, 975: -1, 976: -1, 977: -1, 978: -1, 979: -1, 980: 194, 981: 195, 982: -1, 983: -1, 984: 196, 985: -1, 986: 197, 987: 198, 988: 199, 989: -1, 990: -1, 991: -1, 992: -1, 993: -1, 994: -1, 995: -1, 996: -1, 997: -1, 998: -1, 999: -1}
# For ImageNet-A 200 categories
    imagenet_a_mask = [k for k in thousand_k_to_200 if thousand_k_to_200[k] != -1]
    print(imagenet_a_mask)


    imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop",
                        "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]

    imagenet_template = 'a photo of a {}.'
    

    set_random_seed(0)
    checkpoint = "openai/clip-vit-large-patch14-336"
    model1 = CLIPModel.from_pretrained(checkpoint)
    processor = CLIPProcessor.from_pretrained(checkpoint)

    # Move model to GPU if available
    device = torch.device("cuda:6")
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    model1.to(device)

    cache_dir = '/home/cxc/TPT/ood_dataset'
    dataset_name = 'barkermrl/imagenet-a'
    num_samples_per_class = 4
    dataset_loading(dataset_name, num_samples_per_class, cache_dir)
    # Load the dataset from disk
    OOD_dataset = load_from_disk(cache_dir)

    imagenet_a_classes = [imagenet_classes[i] for i in imagenet_a_mask]

    imagenet_a_prompts = [imagenet_template.format(
        label) for label in imagenet_a_classes]  #构建prompts

    # Calculate accuracy and confidence
    acc, test_confidence, test_correct = get_clip_results(
        OOD_dataset, model1, imagenet_a_prompts)

    print('ImageNet-A Accuracy (%):', round(100 * acc, 4))

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cudnn.benchmark = True

    classnames = imagenet_classes
    n_ctx = 4
    ctx_init = "a_photo_of_a"
    learning_rate = 5e-3
    context_length = 77
    
    
    
    # Load CLIP encoders
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(checkpoint)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(checkpoint)
    tokenizer = CLIPTokenizer.from_pretrained(checkpoint)
    clip_model = CLIPModel.from_pretrained(checkpoint)

    logit_scale = clip_model.logit_scale


    model = TestTimePromptTuning(device, image_encoder, text_encoder, tokenizer,
                                logit_scale, context_length, classnames, None, n_ctx=n_ctx,
                                ctx_init=ctx_init)

    model_state = None

    # Freeze all parameters except for those in PromptLearner
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad = False

    # Verify that the correct parameters are set to require gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter requiring gradients: {name}")


    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, learning_rate)
    optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    print('=> Using native Torch AMP. Training in mixed precision.')


    classnames_all = imagenet_classes
    classnames = [classnames_all[i] for i in imagenet_a_mask]
    model.reset_classnames(classnames, tokenizer, text_encoder)

    OOD_dataset.set_transform(transform_function)
    batchsize = 1
    
    data_loader = DataLoader(OOD_dataset, batch_size=batchsize,
                         shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=4)
    
    
    acc, test_confidence, test_correct = test_time_adapt_eval(data_loader, model, optimizer,
                                                          optim_state, scaler, device)