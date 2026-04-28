---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:4383
- loss:BinaryCrossEntropyLoss
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model trained using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
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
    ['Công dân có quyền tiếp cận thông tin thuộc bí mật nhà nước ngay lập tức sau khi thông tin được giải mật?', 'Thông tin công dân được tiếp cận có điều kiện\n\n1. Thông tin liên quan đến bí mật kinh doanh được tiếp cận trong trường hợp chủ sở hữu bí mật kinh doanh đó đồng ý.\n\n2. Thông tin liên quan đến bí mật đời sống riêng tư, bí mật cá nhân được tiếp cận trong trường hợp được người đó đồng ý; thông tin liên quan đến bí mật gia đình được tiếp cận trong trường hợp được các thành viên gia đình đồng ý.\n\n3. Trong quá trình thực hiện chức năng, nhiệm vụ, quyền hạn của mình, người đứng đầu cơ quan nhà nước quyết định việc cung cấp thông tin liên quan đến bí mật kinh doanh, đời sống riêng tư, bí mật cá nhân, bí mật gia đình trong trường hợp cần thiết vì lợi ích công cộng, sức khỏe của cộng đồng theo quy định của luật có liên quan mà không cần có sự đồng ý theo quy định tại khoản 1 và khoản 2 Điều này. Thông tin công dân được tiếp cận có điều kiện 1. Thông tin liên quan đến bí mật kinh doanh được tiếp cận trong trường hợp chủ sở hữu bí mật kinh doanh đó đồng ý. 1. Thông tin liên quan đến bí mật kinh doanh được tiếp cận trong trường hợp chủ sở hữu bí mật kinh doanh đó đồng ý. 2. Thông tin liên quan đến bí mật đời sống riêng tư, bí mật cá nhân được tiếp cận trong trường hợp được người đó đồng ý; thông tin liên quan đến bí mật gia đình được tiếp cận trong trường hợp được các thành viên gia đình đồng ý. 2. Thông tin liên quan đến bí mật đời sống riêng tư, bí mật cá nhân được tiếp cận trong trường hợp được người đó đồng ý; thông tin liên quan đến bí mật gia đình được tiếp cận trong trường hợp được các thành viên gia đình đồng ý. 3. Trong quá trình thực hiện chức năng, nhiệm vụ, quyền hạn của mình, người đứng đầu cơ quan nhà nước quyết định việc cung cấp thông tin liên quan đến bí mật kinh doanh, đời sống riêng tư, bí mật cá nhân, bí mật gia đình trong trường hợp cần thiết vì lợi ích công cộng, sức khỏe của cộng đồng theo quy định của luật có liên quan mà không cần có sự đồng ý theo quy định tại khoản 1 và khoản 2 Điều này. 3. Trong quá trình thực hiện chức năng, nhiệm vụ, quyền hạn của mình, người đứng đầu cơ quan nhà nước quyết định việc cung cấp thông tin liên quan đến bí mật kinh doanh, đời sống riêng tư, bí mật cá nhân, bí mật gia đình trong trường hợp cần thiết vì lợi ích công cộng, sức khỏe của cộng đồng theo quy định của luật có liên quan mà không cần có sự đồng ý theo quy định tại khoản 1 và khoản 2 Điều này. Thông tin công dân được tiếp cận có điều kiện Thông tin liên quan đến bí mật kinh doanh được tiếp cận trong trường hợp chủ sở hữu bí mật kinh doanh đó đồng ý. Thông tin liên quan đến bí mật đời sống riêng tư, bí mật cá nhân được tiếp cận trong trường hợp được người đó đồng ý; thông tin liên quan đến bí mật gia đình được tiếp cận trong trường hợp được các thành viên gia đình đồng ý. Trong quá trình thực hiện chức năng, nhiệm vụ, quyền hạn của mình, người đứng đầu cơ quan nhà nước quyết định việc cung cấp thông tin liên quan đến bí mật kinh doanh, đời sống riêng tư, bí mật cá nhân, bí mật gia đình trong trường hợp cần thiết vì lợi ích công cộng, sức khỏe của cộng đồng theo quy định của luật có liên quan mà không cần có sự đồng ý theo quy định tại khoản 1 và khoản 2 Điều này.'],
    ['Chủ tịch Ủy ban nhân dân cấp huyện ra quyết định và tổ chức quản lý, hỗ trợ xã hội sau cai nghiện ma túy, đúng hay sai?', 'Cai nghiện ma túy tự nguyện tại gia đình, cộng đồng\n\n1. Cai nghiện ma túy tự nguyện tại gia đình, cộng đồng là việc người nghiện ma túy thực hiện cai nghiện tự nguyện tại gia đình, cộng đồng với sự hỗ trợ chuyên môn của tổ chức, cá nhân cung cấp dịch vụ cai nghiện ma túy, sự phối hợp, trợ giúp của gia đình, cộng đồng và chịu sự quản lý của Ủy ban nhân dân cấp xã.\n\n2. Thời hạn cai nghiện ma túy tự nguyện tại gia đình, cộng đồng là từ đủ 06 tháng đến 12 tháng.\n\n3. Người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng khi hoàn thành ít nhất 03 giai đoạn quy định tại các điểm a, b và c khoản 1 Điều 29 của Luật này được hỗ trợ kinh phí.\n\n4. Người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng có trách nhiệm sau đây:\n\na) Thực hiện đúng, đầy đủ các quy định về cai nghiện ma túy tự nguyện và tuân thủ hướng dẫn của cơ quan chuyên môn;\n\nb) Nộp chi phí liên quan đến cai nghiện ma túy theo quy định.\n\n5. Chủ tịch Ủy ban nhân dân cấp xã có trách nhiệm sau đây:\n\na) Tiếp nhận đăng ký cai nghiện ma túy tự nguyện tại gia đình, cộng đồng;\n\nb) Hướng dẫn, quản lý người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng;\n\nc) Cấp giấy xác nhận hoàn thành cai nghiện ma túy tự nguyện tại gia đình, cộng đồng.\n\n6. Chủ tịch Ủy ban nhân dân cấp huyện có trách nhiệm sau đây:\n\na) Giao nhiệm vụ cho các đơn vị sự nghiệp công lập thuộc thẩm quyền trên địa bàn cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng;\n\nb) Tiếp nhận đăng ký và công bố danh sách tổ chức, cá nhân đủ điều kiện cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng;\n\nc) Thông báo cho Ủy ban nhân dân cấp xã danh sách tổ chức, cá nhân cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng;\n\nd) Bố trí kinh phí hỗ trợ công tác cai nghiện ma túy tự nguyện tại gia đình, cộng đồng;\n\nđ) Chỉ đạo, hướng dẫn, kiểm tra công tác cai nghiện ma túy tự nguyện tại gia đình, cộng đồng.\n\n7. Cơ sở cai nghiện ma túy, tổ chức, cá nhân đủ điều kiện cung cấp một hoặc nhiều hoạt động cai nghiện theo quy trình cai nghiện ma túy quy định tại khoản 1 Điều 29 của Luật này được cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng và có trách nhiệm sau đây:\n\na) Tiếp nhận và tổ chức thực hiện cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng;\n\nb) Thực hiện đúng quy trình chuyên môn nghiệp vụ theo quy định của cơ quan có thẩm quyền;\n\nc) Trong thời hạn 05 ngày làm việc kể từ ngày người cai nghiện ma túy sử dụng dịch vụ hoặc tự ý chấm dứt việc sử dụng dịch vụ hoặc hoàn thành dịch vụ phải thông báo cho Ủy ban nhân dân cấp xã nơi người đó đăng ký cai nghiện ma túy tự nguyện tại gia đình, cộng đồng.\n\n8. Tổ chức, cá nhân có đủ điều kiện thì được đăng ký cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng với Chủ tịch Ủy ban nhân dân cấp huyện.\n\n9. Chính phủ quy định chi tiết Điều này. Cai nghiện ma túy tự nguyện tại gia đình, cộng đồng 1. Cai nghiện ma túy tự nguyện tại gia đình, cộng đồng là việc người nghiện ma túy thực hiện cai nghiện tự nguyện tại gia đình, cộng đồng với sự hỗ trợ chuyên môn của tổ chức, cá nhân cung cấp dịch vụ cai nghiện ma túy, sự phối hợp, trợ giúp của gia đình, cộng đồng và chịu sự quản lý của Ủy ban nhân dân cấp xã. 1. Cai nghiện ma túy tự nguyện tại gia đình, cộng đồng là việc người nghiện ma túy thực hiện cai nghiện tự nguyện tại gia đình, cộng đồng với sự hỗ trợ chuyên môn của tổ chức, cá nhân cung cấp dịch vụ cai nghiện ma túy, sự phối hợp, trợ giúp của gia đình, cộng đồng và chịu sự quản lý của Ủy ban nhân dân cấp xã. 2. Thời hạn cai nghiện ma túy tự nguyện tại gia đình, cộng đồng là từ đủ 06 tháng đến 12 tháng. 2. Thời hạn cai nghiện ma túy tự nguyện tại gia đình, cộng đồng là từ đủ 06 tháng đến 12 tháng. 3. Người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng khi hoàn thành ít nhất 03 giai đoạn quy định tại các điểm a, b và c khoản 1 Điều 29 của Luật này được hỗ trợ kinh phí. 3. Người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng khi hoàn thành ít nhất 03 giai đoạn quy định tại các điểm a, b và c khoản 1 Điều 29 của Luật này được hỗ trợ kinh phí. 4. Người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng có trách nhiệm sau đây: 4. Người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng có trách nhiệm sau đây: a) Thực hiện đúng, đầy đủ các quy định về cai nghiện ma túy tự nguyện và tuân thủ hướng dẫn của cơ quan chuyên môn; a) Thực hiện đúng, đầy đủ các quy định về cai nghiện ma túy tự nguyện và tuân thủ hướng dẫn của cơ quan chuyên môn; b) Nộp chi phí liên quan đến cai nghiện ma túy theo quy định. b) Nộp chi phí liên quan đến cai nghiện ma túy theo quy định. 5. Chủ tịch Ủy ban nhân dân cấp xã có trách nhiệm sau đây: 5. Chủ tịch Ủy ban nhân dân cấp xã có trách nhiệm sau đây: a) Tiếp nhận đăng ký cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; a) Tiếp nhận đăng ký cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; b) Hướng dẫn, quản lý người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; b) Hướng dẫn, quản lý người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; c) Cấp giấy xác nhận hoàn thành cai nghiện ma túy tự nguyện tại gia đình, cộng đồng. c) Cấp giấy xác nhận hoàn thành cai nghiện ma túy tự nguyện tại gia đình, cộng đồng. 6. Chủ tịch Ủy ban nhân dân cấp huyện có trách nhiệm sau đây: 6. Chủ tịch Ủy ban nhân dân cấp huyện có trách nhiệm sau đây: a) Giao nhiệm vụ cho các đơn vị sự nghiệp công lập thuộc thẩm quyền trên địa bàn cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; a) Giao nhiệm vụ cho các đơn vị sự nghiệp công lập thuộc thẩm quyền trên địa bàn cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; b) Tiếp nhận đăng ký và công bố danh sách tổ chức, cá nhân đủ điều kiện cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; b) Tiếp nhận đăng ký và công bố danh sách tổ chức, cá nhân đủ điều kiện cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; c) Thông báo cho Ủy ban nhân dân cấp xã danh sách tổ chức, cá nhân cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; c) Thông báo cho Ủy ban nhân dân cấp xã danh sách tổ chức, cá nhân cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; d) Bố trí kinh phí hỗ trợ công tác cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; d) Bố trí kinh phí hỗ trợ công tác cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; đ) Chỉ đạo, hướng dẫn, kiểm tra công tác cai nghiện ma túy tự nguyện tại gia đình, cộng đồng. đ) Chỉ đạo, hướng dẫn, kiểm tra công tác cai nghiện ma túy tự nguyện tại gia đình, cộng đồng. 7. Cơ sở cai nghiện ma túy, tổ chức, cá nhân đủ điều kiện cung cấp một hoặc nhiều hoạt động cai nghiện theo quy trình cai nghiện ma túy quy định tại khoản 1 Điều 29 của Luật này được cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng và có trách nhiệm sau đây: 7. Cơ sở cai nghiện ma túy, tổ chức, cá nhân đủ điều kiện cung cấp một hoặc nhiều hoạt động cai nghiện theo quy trình cai nghiện ma túy quy định tại khoản 1 Điều 29 của Luật này được cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng và có trách nhiệm sau đây: a) Tiếp nhận và tổ chức thực hiện cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; a) Tiếp nhận và tổ chức thực hiện cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; b) Thực hiện đúng quy trình chuyên môn nghiệp vụ theo quy định của cơ quan có thẩm quyền; b) Thực hiện đúng quy trình chuyên môn nghiệp vụ theo quy định của cơ quan có thẩm quyền; c) Trong thời hạn 05 ngày làm việc kể từ ngày người cai nghiện ma túy sử dụng dịch vụ hoặc tự ý chấm dứt việc sử dụng dịch vụ hoặc hoàn thành dịch vụ phải thông báo cho Ủy ban nhân dân cấp xã nơi người đó đăng ký cai nghiện ma túy tự nguyện tại gia đình, cộng đồng. c) Trong thời hạn 05 ngày làm việc kể từ ngày người cai nghiện ma túy sử dụng dịch vụ hoặc tự ý chấm dứt việc sử dụng dịch vụ hoặc hoàn thành dịch vụ phải thông báo cho Ủy ban nhân dân cấp xã nơi người đó đăng ký cai nghiện ma túy tự nguyện tại gia đình, cộng đồng. 8. Tổ chức, cá nhân có đủ điều kiện thì được đăng ký cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng với Chủ tịch Ủy ban nhân dân cấp huyện. 8. Tổ chức, cá nhân có đủ điều kiện thì được đăng ký cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng với Chủ tịch Ủy ban nhân dân cấp huyện. 9. Chính phủ quy định chi tiết Điều này. 9. Chính phủ quy định chi tiết Điều này. Cai nghiện ma túy tự nguyện tại gia đình, cộng đồng Cai nghiện ma túy tự nguyện tại gia đình, cộng đồng là việc người nghiện ma túy thực hiện cai nghiện tự nguyện tại gia đình, cộng đồng với sự hỗ trợ chuyên môn của tổ chức, cá nhân cung cấp dịch vụ cai nghiện ma túy, sự phối hợp, trợ giúp của gia đình, cộng đồng và chịu sự quản lý của Ủy ban nhân dân cấp xã. Thời hạn cai nghiện ma túy tự nguyện tại gia đình, cộng đồng là từ đủ 06 tháng đến 12 tháng. Người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng khi hoàn thành ít nhất 03 giai đoạn quy định tại các điểm a, b và c khoản 1 Điều 29 của Luật này được hỗ trợ kinh phí. Người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng có trách nhiệm sau đây: a) Thực hiện đúng, đầy đủ các quy định về cai nghiện ma túy tự nguyện và tuân thủ hướng dẫn của cơ quan chuyên môn; b) Nộp chi phí liên quan đến cai nghiện ma túy theo quy định. Chủ tịch Ủy ban nhân dân cấp xã có trách nhiệm sau đây: a) Tiếp nhận đăng ký cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; b) Hướng dẫn, quản lý người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; c) Cấp giấy xác nhận hoàn thành cai nghiện ma túy tự nguyện tại gia đình, cộng đồng. Chủ tịch Ủy ban nhân dân cấp huyện có trách nhiệm sau đây: a) Giao nhiệm vụ cho các đơn vị sự nghiệp công lập thuộc thẩm quyền trên địa bàn cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; b) Tiếp nhận đăng ký và công bố danh sách tổ chức, cá nhân đủ điều kiện cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; c) Thông báo cho Ủy ban nhân dân cấp xã danh sách tổ chức, cá nhân cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; d) Bố trí kinh phí hỗ trợ công tác cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; đ) Chỉ đạo, hướng dẫn, kiểm tra công tác cai nghiện ma túy tự nguyện tại gia đình, cộng đồng. Cơ sở cai nghiện ma túy, tổ chức, cá nhân đủ điều kiện cung cấp một hoặc nhiều hoạt động cai nghiện theo quy trình cai nghiện ma túy quy định tại khoản 1 Điều 29 của Luật này được cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng và có trách nhiệm sau đây: a) Tiếp nhận và tổ chức thực hiện cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; b) Thực hiện đúng quy trình chuyên môn nghiệp vụ theo quy định của cơ quan có thẩm quyền; c) Trong thời hạn 05 ngày làm việc kể từ ngày người cai nghiện ma túy sử dụng dịch vụ hoặc tự ý chấm dứt việc sử dụng dịch vụ hoặc hoàn thành dịch vụ phải thông báo cho Ủy ban nhân dân cấp xã nơi người đó đăng ký cai nghiện ma túy tự nguyện tại gia đình, cộng đồng. Tổ chức, cá nhân có đủ điều kiện thì được đăng ký cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng với Chủ tịch Ủy ban nhân dân cấp huyện. Chính phủ quy định chi tiết Điều này.'],
    ['Thông tin do cơ quan nhà nước tạo ra được định nghĩa như thế nào theo Luật Tiếp cận thông tin?', 'Cách thức tiếp cận thông tin\n\nCông dân được tiếp cận thông tin bằng các cách thức sau:\n\n1. Tự do tiếp cận thông tin được cơ quan nhà nước công khai;\n\n2. Yêu cầu cơ quan nhà nước cung cấp thông tin. Cách thức tiếp cận thông tin Công dân được tiếp cận thông tin bằng các cách thức sau: Công dân được tiếp cận thông tin bằng các cách thức sau: 1. Tự do tiếp cận thông tin được cơ quan nhà nước công khai; 1. Tự do tiếp cận thông tin được cơ quan nhà nước công khai; 2. Yêu cầu cơ quan nhà nước cung cấp thông tin. 2. Yêu cầu cơ quan nhà nước cung cấp thông tin. Cách thức tiếp cận thông tin Công dân được tiếp cận thông tin bằng các cách thức sau: Tự do tiếp cận thông tin được cơ quan nhà nước công khai; Yêu cầu cơ quan nhà nước cung cấp thông tin.'],
    ['Chủ sở hữu, chủ thể có quyền khác đối với tài sản bị thiệt hại trong tình thế cấp thiết được bồi thường thiệt hại bởi người gây ra tình thế cấp thiết, đúng hay sai?', 'Quyền yêu cầu bồi thường thiệt hại\n\nChủ sở hữu, chủ thể có quyền khác đối với tài sản có quyền yêu cầu người có hành vi xâm phạm quyền sở hữu, quyền khác đối với tài sản bồi thường thiệt hại. Quyền yêu cầu bồi thường thiệt hại Chủ sở hữu, chủ thể có quyền khác đối với tài sản có quyền yêu cầu người có hành vi xâm phạm quyền sở hữu, quyền khác đối với tài sản bồi thường thiệt hại. Chủ sở hữu, chủ thể có quyền khác đối với tài sản có quyền yêu cầu người có hành vi xâm phạm quyền sở hữu, quyền khác đối với tài sản bồi thường thiệt hại. Quyền yêu cầu bồi thường thiệt hại Chủ sở hữu, chủ thể có quyền khác đối với tài sản có quyền yêu cầu người có hành vi xâm phạm quyền sở hữu, quyền khác đối với tài sản bồi thường thiệt hại.'],
    ['Nếu không phạm tội quả tang, một người sẽ không bị bắt nếu không có quyết định hoặc phê chuẩn của cơ quan nhà nước có thẩm quyền theo quy định của pháp luật. Đúng hay sai?', 'Điều kiện của nhà ở tham gia giao dịch\n1. Giao dịch về mua bán, thuê mua, tặng cho, đổi, thế chấp, góp vốn bằng nhà ở thì nhà ở phải có đủ điều kiện sau đây:\na) Có Giấy chứng nhận theo quy định của pháp luật, trừ trường hợp quy định tại khoản 2 Điều này;\nb) Không thuộc trường hợp đang có tranh chấp, khiếu nại, khiếu kiện về quyền sở hữu theo quy định của pháp luật về giải quyết tranh chấp, khiếu nại, tố cáo;\nc) Đang trong thời hạn sở hữu nhà ở đối với trường hợp sở hữu nhà ở có thời hạn;\nd) Không bị kê biên để thi hành án hoặc để chấp hành quyết định hành chính đã có hiệu lực pháp luật của cơ quan nhà nước có thẩm quyền hoặc không thuộc trường hợp bị áp dụng biện pháp khẩn cấp tạm thời, biện pháp ngăn chặn theo quyết định của Tòa án hoặc cơ quan nhà nước có thẩm quyền;\nđ) Không thuộc trường hợp đã có quyết định thu hồi đất, có thông báo giải tỏa, phá dỡ nhà ở của cơ quan có thẩm quyền;\ne) Điều kiện quy định tại điểm b và điểm c khoản này không áp dụng đối với trường hợp mua bán, thuê mua nhà ở hình thành trong tương lai.\n2. Giao dịch về nhà ở sau đây thì nhà ở không bắt buộc phải có Giấy chứng nhận:\na) Mua bán, thuê mua, thế chấp nhà ở hình thành trong tương lai; bán nhà ở trong trường hợp giải thể, phá sản;\nb) Tổ chức tặng cho nhà tình nghĩa, nhà tình thương, nhà đại đoàn kết;\nc) Mua bán, thuê mua nhà ở có sẵn của chủ đầu tư dự án đầu tư xây dựng nhà ở trong các trường hợp sau đây: nhà ở thuộc tài sản công; nhà ở xã hội, nhà ở cho lực lượng vũ trang nhân dân, nhà ở phục vụ tái định cư không thuộc tài sản công;\nd) Cho thuê, cho mượn, cho ở nhờ, ủy quyền quản lý nhà ở;\nđ) Nhận thừa kế nhà ở.\nGiấy tờ chứng minh điều kiện nhà ở tham gia giao dịch quy định tại khoản này thực hiện theo quy định của Chính phủ.\n3. Trường hợp nhà ở cho thuê thì ngoài điều kiện quy định tại các điểm c, d và đ khoản 1 Điều này, nhà ở còn phải bảo đảm chất lượng, an toàn cho bên thuê nhà ở, có đầy đủ hệ thống điện, cấp nước, thoát nước, bảo đảm vệ sinh môi trường, trừ trường hợp các bên có thỏa thuận khác. Điều kiện của nhà ở tham gia giao dịch 1. Giao dịch về mua bán, thuê mua, tặng cho, đổi, thế chấp, góp vốn bằng nhà ở thì nhà ở phải có đủ điều kiện sau đây: 1. Giao dịch về mua bán, thuê mua, tặng cho, đổi, thế chấp, góp vốn bằng nhà ở thì nhà ở phải có đủ điều kiện sau đây: a) Có Giấy chứng nhận theo quy định của pháp luật, trừ trường hợp quy định tại khoản 2 Điều này; a) Có Giấy chứng nhận theo quy định của pháp luật, trừ trường hợp quy định tại khoản 2 Điều này; b) Không thuộc trường hợp đang có tranh chấp, khiếu nại, khiếu kiện về quyền sở hữu theo quy định của pháp luật về giải quyết tranh chấp, khiếu nại, tố cáo; b) Không thuộc trường hợp đang có tranh chấp, khiếu nại, khiếu kiện về quyền sở hữu theo quy định của pháp luật về giải quyết tranh chấp, khiếu nại, tố cáo; c) Đang trong thời hạn sở hữu nhà ở đối với trường hợp sở hữu nhà ở có thời hạn; c) Đang trong thời hạn sở hữu nhà ở đối với trường hợp sở hữu nhà ở có thời hạn; d) Không bị kê biên để thi hành án hoặc để chấp hành quyết định hành chính đã có hiệu lực pháp luật của cơ quan nhà nước có thẩm quyền hoặc không thuộc trường hợp bị áp dụng biện pháp khẩn cấp tạm thời, biện pháp ngăn chặn theo quyết định của Tòa án hoặc cơ quan nhà nước có thẩm quyền; d) Không bị kê biên để thi hành án hoặc để chấp hành quyết định hành chính đã có hiệu lực pháp luật của cơ quan nhà nước có thẩm quyền hoặc không thuộc trường hợp bị áp dụng biện pháp khẩn cấp tạm thời, biện pháp ngăn chặn theo quyết định của Tòa án hoặc cơ quan nhà nước có thẩm quyền; đ) Không thuộc trường hợp đã có quyết định thu hồi đất, có thông báo giải tỏa, phá dỡ nhà ở của cơ quan có thẩm quyền; đ) Không thuộc trường hợp đã có quyết định thu hồi đất, có thông báo giải tỏa, phá dỡ nhà ở của cơ quan có thẩm quyền; e) Điều kiện quy định tại điểm b và điểm c khoản này không áp dụng đối với trường hợp mua bán, thuê mua nhà ở hình thành trong tương lai. e) Điều kiện quy định tại điểm b và điểm c khoản này không áp dụng đối với trường hợp mua bán, thuê mua nhà ở hình thành trong tương lai. 2. Giao dịch về nhà ở sau đây thì nhà ở không bắt buộc phải có Giấy chứng nhận: 2. Giao dịch về nhà ở sau đây thì nhà ở không bắt buộc phải có Giấy chứng nhận: a) Mua bán, thuê mua, thế chấp nhà ở hình thành trong tương lai; bán nhà ở trong trường hợp giải thể, phá sản; a) Mua bán, thuê mua, thế chấp nhà ở hình thành trong tương lai; bán nhà ở trong trường hợp giải thể, phá sản; b) Tổ chức tặng cho nhà tình nghĩa, nhà tình thương, nhà đại đoàn kết; b) Tổ chức tặng cho nhà tình nghĩa, nhà tình thương, nhà đại đoàn kết; c) Mua bán, thuê mua nhà ở có sẵn của chủ đầu tư dự án đầu tư xây dựng nhà ở trong các trường hợp sau đây: nhà ở thuộc tài sản công; nhà ở xã hội, nhà ở cho lực lượng vũ trang nhân dân, nhà ở phục vụ tái định cư không thuộc tài sản công; c) Mua bán, thuê mua nhà ở có sẵn của chủ đầu tư dự án đầu tư xây dựng nhà ở trong các trường hợp sau đây: nhà ở thuộc tài sản công; nhà ở xã hội, nhà ở cho lực lượng vũ trang nhân dân, nhà ở phục vụ tái định cư không thuộc tài sản công; d) Cho thuê, cho mượn, cho ở nhờ, ủy quyền quản lý nhà ở; d) Cho thuê, cho mượn, cho ở nhờ, ủy quyền quản lý nhà ở; đ) Nhận thừa kế nhà ở. đ) Nhận thừa kế nhà ở. Giấy tờ chứng minh điều kiện nhà ở tham gia giao dịch quy định tại khoản này thực hiện theo quy định của Chính phủ. Giấy tờ chứng minh điều kiện nhà ở tham gia giao dịch quy định tại khoản này thực hiện theo quy định của Chính phủ. 3. Trường hợp nhà ở cho thuê thì ngoài điều kiện quy định tại các điểm c, d và đ khoản 1 Điều này, nhà ở còn phải bảo đảm chất lượng, an toàn cho bên thuê nhà ở, có đầy đủ hệ thống điện, cấp nước, thoát nước, bảo đảm vệ sinh môi trường, trừ trường hợp các bên có thỏa thuận khác. 3. Trường hợp nhà ở cho thuê thì ngoài điều kiện quy định tại các điểm c, d và đ khoản 1 Điều này, nhà ở còn phải bảo đảm chất lượng, an toàn cho bên thuê nhà ở, có đầy đủ hệ thống điện, cấp nước, thoát nước, bảo đảm vệ sinh môi trường, trừ trường hợp các bên có thỏa thuận khác. Điều kiện của nhà ở tham gia giao dịch Giao dịch về mua bán, thuê mua, tặng cho, đổi, thế chấp, góp vốn bằng nhà ở thì nhà ở phải có đủ điều kiện sau đây: a) Có Giấy chứng nhận theo quy định của pháp luật, trừ trường hợp quy định tại khoản 2 Điều này; b) Không thuộc trường hợp đang có tranh chấp, khiếu nại, khiếu kiện về quyền sở hữu theo quy định của pháp luật về giải quyết tranh chấp, khiếu nại, tố cáo; c) Đang trong thời hạn sở hữu nhà ở đối với trường hợp sở hữu nhà ở có thời hạn; d) Không bị kê biên để thi hành án hoặc để chấp hành quyết định hành chính đã có hiệu lực pháp luật của cơ quan nhà nước có thẩm quyền hoặc không thuộc trường hợp bị áp dụng biện pháp khẩn cấp tạm thời, biện pháp ngăn chặn theo quyết định của Tòa án hoặc cơ quan nhà nước có thẩm quyền; đ) Không thuộc trường hợp đã có quyết định thu hồi đất, có thông báo giải tỏa, phá dỡ nhà ở của cơ quan có thẩm quyền; e) Điều kiện quy định tại điểm b và điểm c khoản này không áp dụng đối với trường hợp mua bán, thuê mua nhà ở hình thành trong tương lai. Giao dịch về nhà ở sau đây thì nhà ở không bắt buộc phải có Giấy chứng nhận: a) Mua bán, thuê mua, thế chấp nhà ở hình thành trong tương lai; bán nhà ở trong trường hợp giải thể, phá sản; b) Tổ chức tặng cho nhà tình nghĩa, nhà tình thương, nhà đại đoàn kết; c) Mua bán, thuê mua nhà ở có sẵn của chủ đầu tư dự án đầu tư xây dựng nhà ở trong các trường hợp sau đây: nhà ở thuộc tài sản công; nhà ở xã hội, nhà ở cho lực lượng vũ trang nhân dân, nhà ở phục vụ tái định cư không thuộc tài sản công; d) Cho thuê, cho mượn, cho ở nhờ, ủy quyền quản lý nhà ở; đ) Nhận thừa kế nhà ở. Giấy tờ chứng minh điều kiện nhà ở tham gia giao dịch quy định tại khoản này thực hiện theo quy định của Chính phủ. Trường hợp nhà ở cho thuê thì ngoài điều kiện quy định tại các điểm c, d và đ khoản 1 Điều này, nhà ở còn phải bảo đảm chất lượng, an toàn cho bên thuê nhà ở, có đầy đủ hệ thống điện, cấp nước, thoát nước, bảo đảm vệ sinh môi trường, trừ trường hợp các bên có thỏa thuận khác.'],
]
scores = model.predict(pairs)
print(scores)
# [0.0004 0.0005 0.0004 0.0004 0.0004]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Công dân có quyền tiếp cận thông tin thuộc bí mật nhà nước ngay lập tức sau khi thông tin được giải mật?',
    [
        'Thông tin công dân được tiếp cận có điều kiện\n\n1. Thông tin liên quan đến bí mật kinh doanh được tiếp cận trong trường hợp chủ sở hữu bí mật kinh doanh đó đồng ý.\n\n2. Thông tin liên quan đến bí mật đời sống riêng tư, bí mật cá nhân được tiếp cận trong trường hợp được người đó đồng ý; thông tin liên quan đến bí mật gia đình được tiếp cận trong trường hợp được các thành viên gia đình đồng ý.\n\n3. Trong quá trình thực hiện chức năng, nhiệm vụ, quyền hạn của mình, người đứng đầu cơ quan nhà nước quyết định việc cung cấp thông tin liên quan đến bí mật kinh doanh, đời sống riêng tư, bí mật cá nhân, bí mật gia đình trong trường hợp cần thiết vì lợi ích công cộng, sức khỏe của cộng đồng theo quy định của luật có liên quan mà không cần có sự đồng ý theo quy định tại khoản 1 và khoản 2 Điều này. Thông tin công dân được tiếp cận có điều kiện 1. Thông tin liên quan đến bí mật kinh doanh được tiếp cận trong trường hợp chủ sở hữu bí mật kinh doanh đó đồng ý. 1. Thông tin liên quan đến bí mật kinh doanh được tiếp cận trong trường hợp chủ sở hữu bí mật kinh doanh đó đồng ý. 2. Thông tin liên quan đến bí mật đời sống riêng tư, bí mật cá nhân được tiếp cận trong trường hợp được người đó đồng ý; thông tin liên quan đến bí mật gia đình được tiếp cận trong trường hợp được các thành viên gia đình đồng ý. 2. Thông tin liên quan đến bí mật đời sống riêng tư, bí mật cá nhân được tiếp cận trong trường hợp được người đó đồng ý; thông tin liên quan đến bí mật gia đình được tiếp cận trong trường hợp được các thành viên gia đình đồng ý. 3. Trong quá trình thực hiện chức năng, nhiệm vụ, quyền hạn của mình, người đứng đầu cơ quan nhà nước quyết định việc cung cấp thông tin liên quan đến bí mật kinh doanh, đời sống riêng tư, bí mật cá nhân, bí mật gia đình trong trường hợp cần thiết vì lợi ích công cộng, sức khỏe của cộng đồng theo quy định của luật có liên quan mà không cần có sự đồng ý theo quy định tại khoản 1 và khoản 2 Điều này. 3. Trong quá trình thực hiện chức năng, nhiệm vụ, quyền hạn của mình, người đứng đầu cơ quan nhà nước quyết định việc cung cấp thông tin liên quan đến bí mật kinh doanh, đời sống riêng tư, bí mật cá nhân, bí mật gia đình trong trường hợp cần thiết vì lợi ích công cộng, sức khỏe của cộng đồng theo quy định của luật có liên quan mà không cần có sự đồng ý theo quy định tại khoản 1 và khoản 2 Điều này. Thông tin công dân được tiếp cận có điều kiện Thông tin liên quan đến bí mật kinh doanh được tiếp cận trong trường hợp chủ sở hữu bí mật kinh doanh đó đồng ý. Thông tin liên quan đến bí mật đời sống riêng tư, bí mật cá nhân được tiếp cận trong trường hợp được người đó đồng ý; thông tin liên quan đến bí mật gia đình được tiếp cận trong trường hợp được các thành viên gia đình đồng ý. Trong quá trình thực hiện chức năng, nhiệm vụ, quyền hạn của mình, người đứng đầu cơ quan nhà nước quyết định việc cung cấp thông tin liên quan đến bí mật kinh doanh, đời sống riêng tư, bí mật cá nhân, bí mật gia đình trong trường hợp cần thiết vì lợi ích công cộng, sức khỏe của cộng đồng theo quy định của luật có liên quan mà không cần có sự đồng ý theo quy định tại khoản 1 và khoản 2 Điều này.',
        'Cai nghiện ma túy tự nguyện tại gia đình, cộng đồng\n\n1. Cai nghiện ma túy tự nguyện tại gia đình, cộng đồng là việc người nghiện ma túy thực hiện cai nghiện tự nguyện tại gia đình, cộng đồng với sự hỗ trợ chuyên môn của tổ chức, cá nhân cung cấp dịch vụ cai nghiện ma túy, sự phối hợp, trợ giúp của gia đình, cộng đồng và chịu sự quản lý của Ủy ban nhân dân cấp xã.\n\n2. Thời hạn cai nghiện ma túy tự nguyện tại gia đình, cộng đồng là từ đủ 06 tháng đến 12 tháng.\n\n3. Người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng khi hoàn thành ít nhất 03 giai đoạn quy định tại các điểm a, b và c khoản 1 Điều 29 của Luật này được hỗ trợ kinh phí.\n\n4. Người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng có trách nhiệm sau đây:\n\na) Thực hiện đúng, đầy đủ các quy định về cai nghiện ma túy tự nguyện và tuân thủ hướng dẫn của cơ quan chuyên môn;\n\nb) Nộp chi phí liên quan đến cai nghiện ma túy theo quy định.\n\n5. Chủ tịch Ủy ban nhân dân cấp xã có trách nhiệm sau đây:\n\na) Tiếp nhận đăng ký cai nghiện ma túy tự nguyện tại gia đình, cộng đồng;\n\nb) Hướng dẫn, quản lý người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng;\n\nc) Cấp giấy xác nhận hoàn thành cai nghiện ma túy tự nguyện tại gia đình, cộng đồng.\n\n6. Chủ tịch Ủy ban nhân dân cấp huyện có trách nhiệm sau đây:\n\na) Giao nhiệm vụ cho các đơn vị sự nghiệp công lập thuộc thẩm quyền trên địa bàn cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng;\n\nb) Tiếp nhận đăng ký và công bố danh sách tổ chức, cá nhân đủ điều kiện cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng;\n\nc) Thông báo cho Ủy ban nhân dân cấp xã danh sách tổ chức, cá nhân cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng;\n\nd) Bố trí kinh phí hỗ trợ công tác cai nghiện ma túy tự nguyện tại gia đình, cộng đồng;\n\nđ) Chỉ đạo, hướng dẫn, kiểm tra công tác cai nghiện ma túy tự nguyện tại gia đình, cộng đồng.\n\n7. Cơ sở cai nghiện ma túy, tổ chức, cá nhân đủ điều kiện cung cấp một hoặc nhiều hoạt động cai nghiện theo quy trình cai nghiện ma túy quy định tại khoản 1 Điều 29 của Luật này được cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng và có trách nhiệm sau đây:\n\na) Tiếp nhận và tổ chức thực hiện cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng;\n\nb) Thực hiện đúng quy trình chuyên môn nghiệp vụ theo quy định của cơ quan có thẩm quyền;\n\nc) Trong thời hạn 05 ngày làm việc kể từ ngày người cai nghiện ma túy sử dụng dịch vụ hoặc tự ý chấm dứt việc sử dụng dịch vụ hoặc hoàn thành dịch vụ phải thông báo cho Ủy ban nhân dân cấp xã nơi người đó đăng ký cai nghiện ma túy tự nguyện tại gia đình, cộng đồng.\n\n8. Tổ chức, cá nhân có đủ điều kiện thì được đăng ký cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng với Chủ tịch Ủy ban nhân dân cấp huyện.\n\n9. Chính phủ quy định chi tiết Điều này. Cai nghiện ma túy tự nguyện tại gia đình, cộng đồng 1. Cai nghiện ma túy tự nguyện tại gia đình, cộng đồng là việc người nghiện ma túy thực hiện cai nghiện tự nguyện tại gia đình, cộng đồng với sự hỗ trợ chuyên môn của tổ chức, cá nhân cung cấp dịch vụ cai nghiện ma túy, sự phối hợp, trợ giúp của gia đình, cộng đồng và chịu sự quản lý của Ủy ban nhân dân cấp xã. 1. Cai nghiện ma túy tự nguyện tại gia đình, cộng đồng là việc người nghiện ma túy thực hiện cai nghiện tự nguyện tại gia đình, cộng đồng với sự hỗ trợ chuyên môn của tổ chức, cá nhân cung cấp dịch vụ cai nghiện ma túy, sự phối hợp, trợ giúp của gia đình, cộng đồng và chịu sự quản lý của Ủy ban nhân dân cấp xã. 2. Thời hạn cai nghiện ma túy tự nguyện tại gia đình, cộng đồng là từ đủ 06 tháng đến 12 tháng. 2. Thời hạn cai nghiện ma túy tự nguyện tại gia đình, cộng đồng là từ đủ 06 tháng đến 12 tháng. 3. Người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng khi hoàn thành ít nhất 03 giai đoạn quy định tại các điểm a, b và c khoản 1 Điều 29 của Luật này được hỗ trợ kinh phí. 3. Người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng khi hoàn thành ít nhất 03 giai đoạn quy định tại các điểm a, b và c khoản 1 Điều 29 của Luật này được hỗ trợ kinh phí. 4. Người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng có trách nhiệm sau đây: 4. Người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng có trách nhiệm sau đây: a) Thực hiện đúng, đầy đủ các quy định về cai nghiện ma túy tự nguyện và tuân thủ hướng dẫn của cơ quan chuyên môn; a) Thực hiện đúng, đầy đủ các quy định về cai nghiện ma túy tự nguyện và tuân thủ hướng dẫn của cơ quan chuyên môn; b) Nộp chi phí liên quan đến cai nghiện ma túy theo quy định. b) Nộp chi phí liên quan đến cai nghiện ma túy theo quy định. 5. Chủ tịch Ủy ban nhân dân cấp xã có trách nhiệm sau đây: 5. Chủ tịch Ủy ban nhân dân cấp xã có trách nhiệm sau đây: a) Tiếp nhận đăng ký cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; a) Tiếp nhận đăng ký cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; b) Hướng dẫn, quản lý người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; b) Hướng dẫn, quản lý người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; c) Cấp giấy xác nhận hoàn thành cai nghiện ma túy tự nguyện tại gia đình, cộng đồng. c) Cấp giấy xác nhận hoàn thành cai nghiện ma túy tự nguyện tại gia đình, cộng đồng. 6. Chủ tịch Ủy ban nhân dân cấp huyện có trách nhiệm sau đây: 6. Chủ tịch Ủy ban nhân dân cấp huyện có trách nhiệm sau đây: a) Giao nhiệm vụ cho các đơn vị sự nghiệp công lập thuộc thẩm quyền trên địa bàn cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; a) Giao nhiệm vụ cho các đơn vị sự nghiệp công lập thuộc thẩm quyền trên địa bàn cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; b) Tiếp nhận đăng ký và công bố danh sách tổ chức, cá nhân đủ điều kiện cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; b) Tiếp nhận đăng ký và công bố danh sách tổ chức, cá nhân đủ điều kiện cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; c) Thông báo cho Ủy ban nhân dân cấp xã danh sách tổ chức, cá nhân cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; c) Thông báo cho Ủy ban nhân dân cấp xã danh sách tổ chức, cá nhân cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; d) Bố trí kinh phí hỗ trợ công tác cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; d) Bố trí kinh phí hỗ trợ công tác cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; đ) Chỉ đạo, hướng dẫn, kiểm tra công tác cai nghiện ma túy tự nguyện tại gia đình, cộng đồng. đ) Chỉ đạo, hướng dẫn, kiểm tra công tác cai nghiện ma túy tự nguyện tại gia đình, cộng đồng. 7. Cơ sở cai nghiện ma túy, tổ chức, cá nhân đủ điều kiện cung cấp một hoặc nhiều hoạt động cai nghiện theo quy trình cai nghiện ma túy quy định tại khoản 1 Điều 29 của Luật này được cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng và có trách nhiệm sau đây: 7. Cơ sở cai nghiện ma túy, tổ chức, cá nhân đủ điều kiện cung cấp một hoặc nhiều hoạt động cai nghiện theo quy trình cai nghiện ma túy quy định tại khoản 1 Điều 29 của Luật này được cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng và có trách nhiệm sau đây: a) Tiếp nhận và tổ chức thực hiện cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; a) Tiếp nhận và tổ chức thực hiện cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; b) Thực hiện đúng quy trình chuyên môn nghiệp vụ theo quy định của cơ quan có thẩm quyền; b) Thực hiện đúng quy trình chuyên môn nghiệp vụ theo quy định của cơ quan có thẩm quyền; c) Trong thời hạn 05 ngày làm việc kể từ ngày người cai nghiện ma túy sử dụng dịch vụ hoặc tự ý chấm dứt việc sử dụng dịch vụ hoặc hoàn thành dịch vụ phải thông báo cho Ủy ban nhân dân cấp xã nơi người đó đăng ký cai nghiện ma túy tự nguyện tại gia đình, cộng đồng. c) Trong thời hạn 05 ngày làm việc kể từ ngày người cai nghiện ma túy sử dụng dịch vụ hoặc tự ý chấm dứt việc sử dụng dịch vụ hoặc hoàn thành dịch vụ phải thông báo cho Ủy ban nhân dân cấp xã nơi người đó đăng ký cai nghiện ma túy tự nguyện tại gia đình, cộng đồng. 8. Tổ chức, cá nhân có đủ điều kiện thì được đăng ký cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng với Chủ tịch Ủy ban nhân dân cấp huyện. 8. Tổ chức, cá nhân có đủ điều kiện thì được đăng ký cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng với Chủ tịch Ủy ban nhân dân cấp huyện. 9. Chính phủ quy định chi tiết Điều này. 9. Chính phủ quy định chi tiết Điều này. Cai nghiện ma túy tự nguyện tại gia đình, cộng đồng Cai nghiện ma túy tự nguyện tại gia đình, cộng đồng là việc người nghiện ma túy thực hiện cai nghiện tự nguyện tại gia đình, cộng đồng với sự hỗ trợ chuyên môn của tổ chức, cá nhân cung cấp dịch vụ cai nghiện ma túy, sự phối hợp, trợ giúp của gia đình, cộng đồng và chịu sự quản lý của Ủy ban nhân dân cấp xã. Thời hạn cai nghiện ma túy tự nguyện tại gia đình, cộng đồng là từ đủ 06 tháng đến 12 tháng. Người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng khi hoàn thành ít nhất 03 giai đoạn quy định tại các điểm a, b và c khoản 1 Điều 29 của Luật này được hỗ trợ kinh phí. Người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng có trách nhiệm sau đây: a) Thực hiện đúng, đầy đủ các quy định về cai nghiện ma túy tự nguyện và tuân thủ hướng dẫn của cơ quan chuyên môn; b) Nộp chi phí liên quan đến cai nghiện ma túy theo quy định. Chủ tịch Ủy ban nhân dân cấp xã có trách nhiệm sau đây: a) Tiếp nhận đăng ký cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; b) Hướng dẫn, quản lý người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; c) Cấp giấy xác nhận hoàn thành cai nghiện ma túy tự nguyện tại gia đình, cộng đồng. Chủ tịch Ủy ban nhân dân cấp huyện có trách nhiệm sau đây: a) Giao nhiệm vụ cho các đơn vị sự nghiệp công lập thuộc thẩm quyền trên địa bàn cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; b) Tiếp nhận đăng ký và công bố danh sách tổ chức, cá nhân đủ điều kiện cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; c) Thông báo cho Ủy ban nhân dân cấp xã danh sách tổ chức, cá nhân cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; d) Bố trí kinh phí hỗ trợ công tác cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; đ) Chỉ đạo, hướng dẫn, kiểm tra công tác cai nghiện ma túy tự nguyện tại gia đình, cộng đồng. Cơ sở cai nghiện ma túy, tổ chức, cá nhân đủ điều kiện cung cấp một hoặc nhiều hoạt động cai nghiện theo quy trình cai nghiện ma túy quy định tại khoản 1 Điều 29 của Luật này được cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng và có trách nhiệm sau đây: a) Tiếp nhận và tổ chức thực hiện cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng; b) Thực hiện đúng quy trình chuyên môn nghiệp vụ theo quy định của cơ quan có thẩm quyền; c) Trong thời hạn 05 ngày làm việc kể từ ngày người cai nghiện ma túy sử dụng dịch vụ hoặc tự ý chấm dứt việc sử dụng dịch vụ hoặc hoàn thành dịch vụ phải thông báo cho Ủy ban nhân dân cấp xã nơi người đó đăng ký cai nghiện ma túy tự nguyện tại gia đình, cộng đồng. Tổ chức, cá nhân có đủ điều kiện thì được đăng ký cung cấp dịch vụ cai nghiện ma túy tự nguyện tại gia đình, cộng đồng với Chủ tịch Ủy ban nhân dân cấp huyện. Chính phủ quy định chi tiết Điều này.',
        'Cách thức tiếp cận thông tin\n\nCông dân được tiếp cận thông tin bằng các cách thức sau:\n\n1. Tự do tiếp cận thông tin được cơ quan nhà nước công khai;\n\n2. Yêu cầu cơ quan nhà nước cung cấp thông tin. Cách thức tiếp cận thông tin Công dân được tiếp cận thông tin bằng các cách thức sau: Công dân được tiếp cận thông tin bằng các cách thức sau: 1. Tự do tiếp cận thông tin được cơ quan nhà nước công khai; 1. Tự do tiếp cận thông tin được cơ quan nhà nước công khai; 2. Yêu cầu cơ quan nhà nước cung cấp thông tin. 2. Yêu cầu cơ quan nhà nước cung cấp thông tin. Cách thức tiếp cận thông tin Công dân được tiếp cận thông tin bằng các cách thức sau: Tự do tiếp cận thông tin được cơ quan nhà nước công khai; Yêu cầu cơ quan nhà nước cung cấp thông tin.',
        'Quyền yêu cầu bồi thường thiệt hại\n\nChủ sở hữu, chủ thể có quyền khác đối với tài sản có quyền yêu cầu người có hành vi xâm phạm quyền sở hữu, quyền khác đối với tài sản bồi thường thiệt hại. Quyền yêu cầu bồi thường thiệt hại Chủ sở hữu, chủ thể có quyền khác đối với tài sản có quyền yêu cầu người có hành vi xâm phạm quyền sở hữu, quyền khác đối với tài sản bồi thường thiệt hại. Chủ sở hữu, chủ thể có quyền khác đối với tài sản có quyền yêu cầu người có hành vi xâm phạm quyền sở hữu, quyền khác đối với tài sản bồi thường thiệt hại. Quyền yêu cầu bồi thường thiệt hại Chủ sở hữu, chủ thể có quyền khác đối với tài sản có quyền yêu cầu người có hành vi xâm phạm quyền sở hữu, quyền khác đối với tài sản bồi thường thiệt hại.',
        'Điều kiện của nhà ở tham gia giao dịch\n1. Giao dịch về mua bán, thuê mua, tặng cho, đổi, thế chấp, góp vốn bằng nhà ở thì nhà ở phải có đủ điều kiện sau đây:\na) Có Giấy chứng nhận theo quy định của pháp luật, trừ trường hợp quy định tại khoản 2 Điều này;\nb) Không thuộc trường hợp đang có tranh chấp, khiếu nại, khiếu kiện về quyền sở hữu theo quy định của pháp luật về giải quyết tranh chấp, khiếu nại, tố cáo;\nc) Đang trong thời hạn sở hữu nhà ở đối với trường hợp sở hữu nhà ở có thời hạn;\nd) Không bị kê biên để thi hành án hoặc để chấp hành quyết định hành chính đã có hiệu lực pháp luật của cơ quan nhà nước có thẩm quyền hoặc không thuộc trường hợp bị áp dụng biện pháp khẩn cấp tạm thời, biện pháp ngăn chặn theo quyết định của Tòa án hoặc cơ quan nhà nước có thẩm quyền;\nđ) Không thuộc trường hợp đã có quyết định thu hồi đất, có thông báo giải tỏa, phá dỡ nhà ở của cơ quan có thẩm quyền;\ne) Điều kiện quy định tại điểm b và điểm c khoản này không áp dụng đối với trường hợp mua bán, thuê mua nhà ở hình thành trong tương lai.\n2. Giao dịch về nhà ở sau đây thì nhà ở không bắt buộc phải có Giấy chứng nhận:\na) Mua bán, thuê mua, thế chấp nhà ở hình thành trong tương lai; bán nhà ở trong trường hợp giải thể, phá sản;\nb) Tổ chức tặng cho nhà tình nghĩa, nhà tình thương, nhà đại đoàn kết;\nc) Mua bán, thuê mua nhà ở có sẵn của chủ đầu tư dự án đầu tư xây dựng nhà ở trong các trường hợp sau đây: nhà ở thuộc tài sản công; nhà ở xã hội, nhà ở cho lực lượng vũ trang nhân dân, nhà ở phục vụ tái định cư không thuộc tài sản công;\nd) Cho thuê, cho mượn, cho ở nhờ, ủy quyền quản lý nhà ở;\nđ) Nhận thừa kế nhà ở.\nGiấy tờ chứng minh điều kiện nhà ở tham gia giao dịch quy định tại khoản này thực hiện theo quy định của Chính phủ.\n3. Trường hợp nhà ở cho thuê thì ngoài điều kiện quy định tại các điểm c, d và đ khoản 1 Điều này, nhà ở còn phải bảo đảm chất lượng, an toàn cho bên thuê nhà ở, có đầy đủ hệ thống điện, cấp nước, thoát nước, bảo đảm vệ sinh môi trường, trừ trường hợp các bên có thỏa thuận khác. Điều kiện của nhà ở tham gia giao dịch 1. Giao dịch về mua bán, thuê mua, tặng cho, đổi, thế chấp, góp vốn bằng nhà ở thì nhà ở phải có đủ điều kiện sau đây: 1. Giao dịch về mua bán, thuê mua, tặng cho, đổi, thế chấp, góp vốn bằng nhà ở thì nhà ở phải có đủ điều kiện sau đây: a) Có Giấy chứng nhận theo quy định của pháp luật, trừ trường hợp quy định tại khoản 2 Điều này; a) Có Giấy chứng nhận theo quy định của pháp luật, trừ trường hợp quy định tại khoản 2 Điều này; b) Không thuộc trường hợp đang có tranh chấp, khiếu nại, khiếu kiện về quyền sở hữu theo quy định của pháp luật về giải quyết tranh chấp, khiếu nại, tố cáo; b) Không thuộc trường hợp đang có tranh chấp, khiếu nại, khiếu kiện về quyền sở hữu theo quy định của pháp luật về giải quyết tranh chấp, khiếu nại, tố cáo; c) Đang trong thời hạn sở hữu nhà ở đối với trường hợp sở hữu nhà ở có thời hạn; c) Đang trong thời hạn sở hữu nhà ở đối với trường hợp sở hữu nhà ở có thời hạn; d) Không bị kê biên để thi hành án hoặc để chấp hành quyết định hành chính đã có hiệu lực pháp luật của cơ quan nhà nước có thẩm quyền hoặc không thuộc trường hợp bị áp dụng biện pháp khẩn cấp tạm thời, biện pháp ngăn chặn theo quyết định của Tòa án hoặc cơ quan nhà nước có thẩm quyền; d) Không bị kê biên để thi hành án hoặc để chấp hành quyết định hành chính đã có hiệu lực pháp luật của cơ quan nhà nước có thẩm quyền hoặc không thuộc trường hợp bị áp dụng biện pháp khẩn cấp tạm thời, biện pháp ngăn chặn theo quyết định của Tòa án hoặc cơ quan nhà nước có thẩm quyền; đ) Không thuộc trường hợp đã có quyết định thu hồi đất, có thông báo giải tỏa, phá dỡ nhà ở của cơ quan có thẩm quyền; đ) Không thuộc trường hợp đã có quyết định thu hồi đất, có thông báo giải tỏa, phá dỡ nhà ở của cơ quan có thẩm quyền; e) Điều kiện quy định tại điểm b và điểm c khoản này không áp dụng đối với trường hợp mua bán, thuê mua nhà ở hình thành trong tương lai. e) Điều kiện quy định tại điểm b và điểm c khoản này không áp dụng đối với trường hợp mua bán, thuê mua nhà ở hình thành trong tương lai. 2. Giao dịch về nhà ở sau đây thì nhà ở không bắt buộc phải có Giấy chứng nhận: 2. Giao dịch về nhà ở sau đây thì nhà ở không bắt buộc phải có Giấy chứng nhận: a) Mua bán, thuê mua, thế chấp nhà ở hình thành trong tương lai; bán nhà ở trong trường hợp giải thể, phá sản; a) Mua bán, thuê mua, thế chấp nhà ở hình thành trong tương lai; bán nhà ở trong trường hợp giải thể, phá sản; b) Tổ chức tặng cho nhà tình nghĩa, nhà tình thương, nhà đại đoàn kết; b) Tổ chức tặng cho nhà tình nghĩa, nhà tình thương, nhà đại đoàn kết; c) Mua bán, thuê mua nhà ở có sẵn của chủ đầu tư dự án đầu tư xây dựng nhà ở trong các trường hợp sau đây: nhà ở thuộc tài sản công; nhà ở xã hội, nhà ở cho lực lượng vũ trang nhân dân, nhà ở phục vụ tái định cư không thuộc tài sản công; c) Mua bán, thuê mua nhà ở có sẵn của chủ đầu tư dự án đầu tư xây dựng nhà ở trong các trường hợp sau đây: nhà ở thuộc tài sản công; nhà ở xã hội, nhà ở cho lực lượng vũ trang nhân dân, nhà ở phục vụ tái định cư không thuộc tài sản công; d) Cho thuê, cho mượn, cho ở nhờ, ủy quyền quản lý nhà ở; d) Cho thuê, cho mượn, cho ở nhờ, ủy quyền quản lý nhà ở; đ) Nhận thừa kế nhà ở. đ) Nhận thừa kế nhà ở. Giấy tờ chứng minh điều kiện nhà ở tham gia giao dịch quy định tại khoản này thực hiện theo quy định của Chính phủ. Giấy tờ chứng minh điều kiện nhà ở tham gia giao dịch quy định tại khoản này thực hiện theo quy định của Chính phủ. 3. Trường hợp nhà ở cho thuê thì ngoài điều kiện quy định tại các điểm c, d và đ khoản 1 Điều này, nhà ở còn phải bảo đảm chất lượng, an toàn cho bên thuê nhà ở, có đầy đủ hệ thống điện, cấp nước, thoát nước, bảo đảm vệ sinh môi trường, trừ trường hợp các bên có thỏa thuận khác. 3. Trường hợp nhà ở cho thuê thì ngoài điều kiện quy định tại các điểm c, d và đ khoản 1 Điều này, nhà ở còn phải bảo đảm chất lượng, an toàn cho bên thuê nhà ở, có đầy đủ hệ thống điện, cấp nước, thoát nước, bảo đảm vệ sinh môi trường, trừ trường hợp các bên có thỏa thuận khác. Điều kiện của nhà ở tham gia giao dịch Giao dịch về mua bán, thuê mua, tặng cho, đổi, thế chấp, góp vốn bằng nhà ở thì nhà ở phải có đủ điều kiện sau đây: a) Có Giấy chứng nhận theo quy định của pháp luật, trừ trường hợp quy định tại khoản 2 Điều này; b) Không thuộc trường hợp đang có tranh chấp, khiếu nại, khiếu kiện về quyền sở hữu theo quy định của pháp luật về giải quyết tranh chấp, khiếu nại, tố cáo; c) Đang trong thời hạn sở hữu nhà ở đối với trường hợp sở hữu nhà ở có thời hạn; d) Không bị kê biên để thi hành án hoặc để chấp hành quyết định hành chính đã có hiệu lực pháp luật của cơ quan nhà nước có thẩm quyền hoặc không thuộc trường hợp bị áp dụng biện pháp khẩn cấp tạm thời, biện pháp ngăn chặn theo quyết định của Tòa án hoặc cơ quan nhà nước có thẩm quyền; đ) Không thuộc trường hợp đã có quyết định thu hồi đất, có thông báo giải tỏa, phá dỡ nhà ở của cơ quan có thẩm quyền; e) Điều kiện quy định tại điểm b và điểm c khoản này không áp dụng đối với trường hợp mua bán, thuê mua nhà ở hình thành trong tương lai. Giao dịch về nhà ở sau đây thì nhà ở không bắt buộc phải có Giấy chứng nhận: a) Mua bán, thuê mua, thế chấp nhà ở hình thành trong tương lai; bán nhà ở trong trường hợp giải thể, phá sản; b) Tổ chức tặng cho nhà tình nghĩa, nhà tình thương, nhà đại đoàn kết; c) Mua bán, thuê mua nhà ở có sẵn của chủ đầu tư dự án đầu tư xây dựng nhà ở trong các trường hợp sau đây: nhà ở thuộc tài sản công; nhà ở xã hội, nhà ở cho lực lượng vũ trang nhân dân, nhà ở phục vụ tái định cư không thuộc tài sản công; d) Cho thuê, cho mượn, cho ở nhờ, ủy quyền quản lý nhà ở; đ) Nhận thừa kế nhà ở. Giấy tờ chứng minh điều kiện nhà ở tham gia giao dịch quy định tại khoản này thực hiện theo quy định của Chính phủ. Trường hợp nhà ở cho thuê thì ngoài điều kiện quy định tại các điểm c, d và đ khoản 1 Điều này, nhà ở còn phải bảo đảm chất lượng, an toàn cho bên thuê nhà ở, có đầy đủ hệ thống điện, cấp nước, thoát nước, bảo đảm vệ sinh môi trường, trừ trường hợp các bên có thỏa thuận khác.',
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

* Size: 4,383 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                             | label                                                          |
  |:--------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                            | string                                                                                 | float                                                          |
  | details | <ul><li>min: 9 tokens</li><li>mean: 32.66 tokens</li><li>max: 96 tokens</li></ul> | <ul><li>min: 23 tokens</li><li>mean: 1248.81 tokens</li><li>max: 8192 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.17</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                           | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:-------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Công dân có quyền tiếp cận thông tin thuộc bí mật nhà nước ngay lập tức sau khi thông tin được giải mật?</code>                | <code>Thông tin công dân được tiếp cận có điều kiện

  1. Thông tin liên quan đến bí mật kinh doanh được tiếp cận trong trường hợp chủ sở hữu bí mật kinh doanh đó đồng ý.

  2. Thông tin liên quan đến bí mật đời sống riêng tư, bí mật cá nhân được tiếp cận trong trường hợp được người đó đồng ý; thông tin liên quan đến bí mật gia đình được tiếp cận trong trường hợp được các thành viên gia đình đồng ý.

  3. Trong quá trình thực hiện chức năng, nhiệm vụ, quyền hạn của mình, người đứng đầu cơ quan nhà nước quyết định việc cung cấp thông tin liên quan đến bí mật kinh doanh, đời sống riêng tư, bí mật cá nhân, bí mật gia đình trong trường hợp cần thiết vì lợi ích công cộng, sức khỏe của cộng đồng theo quy định của luật có liên quan mà không cần có sự đồng ý theo quy định tại khoản 1 và khoản 2 Điều này. Thông tin công dân được tiếp cận có điều kiện 1. Thông tin liên quan đến bí mật kinh doanh được tiếp cận trong trường hợp chủ sở hữu bí mật kinh doanh đó đồng ý. 1. Thông tin liên quan đến bí mật kinh doa...</code> | <code>0.0</code> |
  | <code>Chủ tịch Ủy ban nhân dân cấp huyện ra quyết định và tổ chức quản lý, hỗ trợ xã hội sau cai nghiện ma túy, đúng hay sai?</code> | <code>Cai nghiện ma túy tự nguyện tại gia đình, cộng đồng

  1. Cai nghiện ma túy tự nguyện tại gia đình, cộng đồng là việc người nghiện ma túy thực hiện cai nghiện tự nguyện tại gia đình, cộng đồng với sự hỗ trợ chuyên môn của tổ chức, cá nhân cung cấp dịch vụ cai nghiện ma túy, sự phối hợp, trợ giúp của gia đình, cộng đồng và chịu sự quản lý của Ủy ban nhân dân cấp xã.

  2. Thời hạn cai nghiện ma túy tự nguyện tại gia đình, cộng đồng là từ đủ 06 tháng đến 12 tháng.

  3. Người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng khi hoàn thành ít nhất 03 giai đoạn quy định tại các điểm a, b và c khoản 1 Điều 29 của Luật này được hỗ trợ kinh phí.

  4. Người cai nghiện ma túy tự nguyện tại gia đình, cộng đồng có trách nhiệm sau đây:

  a) Thực hiện đúng, đầy đủ các quy định về cai nghiện ma túy tự nguyện và tuân thủ hướng dẫn của cơ quan chuyên môn;

  b) Nộp chi phí liên quan đến cai nghiện ma túy theo quy định.

  5. Chủ tịch Ủy ban nhân dân cấp xã có trách nhiệm sau đây:

  a) Tiếp nhận đăng ký cai nghi...</code> | <code>0.0</code> |
  | <code>Thông tin do cơ quan nhà nước tạo ra được định nghĩa như thế nào theo Luật Tiếp cận thông tin?</code>                          | <code>Cách thức tiếp cận thông tin<br><br>Công dân được tiếp cận thông tin bằng các cách thức sau:<br><br>1. Tự do tiếp cận thông tin được cơ quan nhà nước công khai;<br><br>2. Yêu cầu cơ quan nhà nước cung cấp thông tin. Cách thức tiếp cận thông tin Công dân được tiếp cận thông tin bằng các cách thức sau: Công dân được tiếp cận thông tin bằng các cách thức sau: 1. Tự do tiếp cận thông tin được cơ quan nhà nước công khai; 1. Tự do tiếp cận thông tin được cơ quan nhà nước công khai; 2. Yêu cầu cơ quan nhà nước cung cấp thông tin. 2. Yêu cầu cơ quan nhà nước cung cấp thông tin. Cách thức tiếp cận thông tin Công dân được tiếp cận thông tin bằng các cách thức sau: Tự do tiếp cận thông tin được cơ quan nhà nước công khai; Yêu cầu cơ quan nhà nước cung cấp thông tin.</code>                                                                                                                                                                                                                                               | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 2
- `num_train_epochs`: 2
- `per_device_eval_batch_size`: 2

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 2
- `num_train_epochs`: 2
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
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
- `per_device_eval_batch_size`: 2
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
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.2281 | 500  | 0.3474        |
| 0.4562 | 1000 | 0.2954        |
| 0.6843 | 1500 | 0.2269        |
| 0.9124 | 2000 | 0.2562        |
| 1.1405 | 2500 | 0.1590        |
| 1.3686 | 3000 | 0.0939        |
| 1.5967 | 3500 | 0.0631        |
| 1.8248 | 4000 | 0.1499        |


### Training Time
- **Training**: 51.2 minutes

### Framework Versions
- Python: 3.10.20
- Sentence Transformers: 5.4.1
- Transformers: 5.5.4
- PyTorch: 2.10.0+cu128
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