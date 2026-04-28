---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:19710
- loss:BinaryCrossEntropyLoss
base_model: AITeamVN/Vietnamese_Reranker
pipeline_tag: text-ranking
library_name: sentence-transformers
metrics:
- accuracy
- accuracy_threshold
- f1
- f1_threshold
- precision
- recall
- average_precision
model-index:
- name: CrossEncoder based on AITeamVN/Vietnamese_Reranker
  results:
  - task:
      type: cross-encoder-binary-classification
      name: Cross Encoder Binary Classification
    dataset:
      name: alqac val
      type: alqac-val
    metrics:
    - type: accuracy
      value: 0.8819444444444444
      name: Accuracy
    - type: accuracy_threshold
      value: 0.9786288738250732
      name: Accuracy Threshold
    - type: f1
      value: 0.7891714520098441
      name: F1
    - type: f1_threshold
      value: 0.030510347336530685
      name: F1 Threshold
    - type: precision
      value: 0.7696
      name: Precision
    - type: recall
      value: 0.8097643097643098
      name: Recall
    - type: average_precision
      value: 0.8579097552979584
      name: Average Precision
---

# CrossEncoder based on AITeamVN/Vietnamese_Reranker

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [AITeamVN/Vietnamese_Reranker](https://huggingface.co/AITeamVN/Vietnamese_Reranker) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [AITeamVN/Vietnamese_Reranker](https://huggingface.co/AITeamVN/Vietnamese_Reranker) <!-- at revision f536976248403314225d7fdfdbc87f0e9516a54e -->
- **Maximum Sequence Length:** 8192 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'XLMRobertaForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['Theo Điều 143, luật Tố tụng hành chính 2015, Tòa án phải đình chỉ vụ án nếu đương sự đã được triệu tập không có mặt, đúng hay sai?', '2. Tòa án triệu tập hợp lệ lần thứ hai, đương sự hoặc người đại diện của họ, người bảo vệ quyền và lợi ích hợp pháp của đương sự phải có mặt tại phiên tòa, nếu vắng mặt không vì sự kiện bất khả kháng, trở ngại khách quan thì xử lý như sau: a) Đối với người khởi kiện, người đại diện theo pháp luật của người khởi kiện mà không có người đại diện tham gia phiên tòa thì bị coi là từ bỏ việc khởi kiện và Tòa án ra quyết định đình chỉ giải quyết vụ án đối với yêu cầu khởi kiện của người đó, trừ trường hợp họ có đơn đề nghị xét xử vắng mặt. Người khởi kiện có quyền khởi kiện lại, nếu thời hiệu khởi kiện vẫn còn;'],
    ['Trường hợp nào sau đây có thể hạn chế quyền tiếp cận thông tin?', 'Công dân được tiếp cận thông tin của cơ quan nhà nước, trừ thông tin không được tiếp cận quy định tại Điều 6 của Luật này; được tiếp cận có điều kiện đối với thông tin quy định tại Điều 7 của Luật này.'],
    ['Người có khó khăn trong nhận thức, làm chủ hành vi phải tự thực hiện yêu cầu cung cấp thông tin, đúng hay sai?', 'c) Người mất năng lực hành vi dân sự; d) Người có khó khăn trong nhận thức, làm chủ hành vi.'],
    ['Độ tuổi con có thể tự mình quản lý tài sản riêng là khi nào ?', 'Quyền có tài sản riêng của con 1. Con có quyền có tài sản riêng. Tài sản riêng của con bao gồm tài sản được thừa kế riêng, được tặng cho riêng, thu nhập do lao động của con, hoa lợi, lợi tức phát sinh từ tài sản riêng của con và thu nhập hợp pháp khác. Tài sản được hình thành từ tài sản riêng của con cũng là tài sản riêng của con.'],
    ['Người tố cáo có quyền yêu cầu được giữ bí mật địa chỉ của mình', 'Trường hợp đặc biệt cần giữ bí mật nhà nước, giữ gìn thuần phong, mỹ tục của dân tộc, giữ bí mật nghề nghiệp, bí mật kinh doanh, bí mật cá nhân hoặc để bảo vệ người chưa thành niên theo yêu cầu của đương sự thì Hội đồng xét xử không công bố các tài liệu có trong hồ sơ vụ án.'],
]
scores = model.predict(pairs)
print(scores)
# [0.0007 0.0006 0.0007 0.0007 0.0007]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Theo Điều 143, luật Tố tụng hành chính 2015, Tòa án phải đình chỉ vụ án nếu đương sự đã được triệu tập không có mặt, đúng hay sai?',
    [
        '2. Tòa án triệu tập hợp lệ lần thứ hai, đương sự hoặc người đại diện của họ, người bảo vệ quyền và lợi ích hợp pháp của đương sự phải có mặt tại phiên tòa, nếu vắng mặt không vì sự kiện bất khả kháng, trở ngại khách quan thì xử lý như sau: a) Đối với người khởi kiện, người đại diện theo pháp luật của người khởi kiện mà không có người đại diện tham gia phiên tòa thì bị coi là từ bỏ việc khởi kiện và Tòa án ra quyết định đình chỉ giải quyết vụ án đối với yêu cầu khởi kiện của người đó, trừ trường hợp họ có đơn đề nghị xét xử vắng mặt. Người khởi kiện có quyền khởi kiện lại, nếu thời hiệu khởi kiện vẫn còn;',
        'Công dân được tiếp cận thông tin của cơ quan nhà nước, trừ thông tin không được tiếp cận quy định tại Điều 6 của Luật này; được tiếp cận có điều kiện đối với thông tin quy định tại Điều 7 của Luật này.',
        'c) Người mất năng lực hành vi dân sự; d) Người có khó khăn trong nhận thức, làm chủ hành vi.',
        'Quyền có tài sản riêng của con 1. Con có quyền có tài sản riêng. Tài sản riêng của con bao gồm tài sản được thừa kế riêng, được tặng cho riêng, thu nhập do lao động của con, hoa lợi, lợi tức phát sinh từ tài sản riêng của con và thu nhập hợp pháp khác. Tài sản được hình thành từ tài sản riêng của con cũng là tài sản riêng của con.',
        'Trường hợp đặc biệt cần giữ bí mật nhà nước, giữ gìn thuần phong, mỹ tục của dân tộc, giữ bí mật nghề nghiệp, bí mật kinh doanh, bí mật cá nhân hoặc để bảo vệ người chưa thành niên theo yêu cầu của đương sự thì Hội đồng xét xử không công bố các tài liệu có trong hồ sơ vụ án.',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Cross Encoder Binary Classification

* Dataset: `alqac-val`
* Evaluated with [<code>CEBinaryClassificationEvaluator</code>](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CEBinaryClassificationEvaluator)

| Metric                | Value      |
|:----------------------|:-----------|
| accuracy              | 0.8819     |
| accuracy_threshold    | 0.9786     |
| f1                    | 0.7892     |
| f1_threshold          | 0.0305     |
| precision             | 0.7696     |
| recall                | 0.8098     |
| **average_precision** | **0.8579** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 19,710 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                          | label                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              | float                                                          |
  | details | <ul><li>min: 8 tokens</li><li>mean: 33.58 tokens</li><li>max: 100 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 93.95 tokens</li><li>max: 1390 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.23</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                      | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | label            |
  |:------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Theo Điều 143, luật Tố tụng hành chính 2015, Tòa án phải đình chỉ vụ án nếu đương sự đã được triệu tập không có mặt, đúng hay sai?</code> | <code>2. Tòa án triệu tập hợp lệ lần thứ hai, đương sự hoặc người đại diện của họ, người bảo vệ quyền và lợi ích hợp pháp của đương sự phải có mặt tại phiên tòa, nếu vắng mặt không vì sự kiện bất khả kháng, trở ngại khách quan thì xử lý như sau: a) Đối với người khởi kiện, người đại diện theo pháp luật của người khởi kiện mà không có người đại diện tham gia phiên tòa thì bị coi là từ bỏ việc khởi kiện và Tòa án ra quyết định đình chỉ giải quyết vụ án đối với yêu cầu khởi kiện của người đó, trừ trường hợp họ có đơn đề nghị xét xử vắng mặt. Người khởi kiện có quyền khởi kiện lại, nếu thời hiệu khởi kiện vẫn còn;</code> | <code>0.0</code> |
  | <code>Trường hợp nào sau đây có thể hạn chế quyền tiếp cận thông tin?</code>                                                                    | <code>Công dân được tiếp cận thông tin của cơ quan nhà nước, trừ thông tin không được tiếp cận quy định tại Điều 6 của Luật này; được tiếp cận có điều kiện đối với thông tin quy định tại Điều 7 của Luật này.</code>                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>0.0</code> |
  | <code>Người có khó khăn trong nhận thức, làm chủ hành vi phải tự thực hiện yêu cầu cung cấp thông tin, đúng hay sai?</code>                     | <code>c) Người mất năng lực hành vi dân sự; d) Người có khó khăn trong nhận thức, làm chủ hành vi.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 4
- `num_train_epochs`: 10
- `per_device_eval_batch_size`: 4

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 4
- `num_train_epochs`: 10
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `per_device_eval_batch_size`: 4
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss | alqac-val_average_precision |
|:------:|:----:|:-------------:|:---------------------------:|
| 0.1015 | 500  | 0.5822        | 0.8169                      |
| 0.2029 | 1000 | 0.4766        | 0.8228                      |
| 0.3044 | 1500 | 0.4936        | 0.8337                      |
| 0.4058 | 2000 | 0.4187        | 0.8189                      |
| 0.5073 | 2500 | 0.4419        | 0.8269                      |
| 0.6088 | 3000 | 0.3684        | 0.8537                      |
| 0.7102 | 3500 | 0.3886        | 0.8283                      |
| 0.8117 | 4000 | 0.3684        | 0.8216                      |
| 0.9131 | 4500 | 0.3408        | 0.8395                      |
| 1.0    | 4928 | -             | 0.8130                      |
| 1.0146 | 5000 | 0.3292        | 0.8245                      |
| 1.1161 | 5500 | 0.2318        | 0.8481                      |
| 1.2175 | 6000 | 0.2713        | 0.8579                      |


### Training Time
- **Training**: 20.5 minutes
- **Evaluation**: 3.0 minutes
- **Total**: 23.5 minutes

### Framework Versions
- Python: 3.10.20
- Sentence Transformers: 5.4.1
- Transformers: 5.5.4
- PyTorch: 2.6.0+cu124
- Accelerate: 1.13.0
- Datasets: 4.8.4
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->