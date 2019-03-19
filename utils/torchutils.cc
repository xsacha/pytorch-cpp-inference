#include "torchutils.h"

// Convert a vector of images to torch Tensor
torch::Tensor __convert_image_to_tensor(const cv::Mat& image)
{
  std::vector<int64_t> dims = {image.rows, image.cols, 3};
  std::vector<int64_t> permute_dims = {2, 0, 1};
  torch::TensorOptions options(torch::kFloat32);
  torch::Tensor image_as_tensor = torch::from_blob(image.data, dims, options);
  image_as_tensor = image_as_tensor.permute(permute_dims);
  return torch::unsqueeze(image_as_tensor, 0);
}

// Predict
std::vector<torch::Tensor> __predict(std::shared_ptr<torch::jit::script::Module> model, torch::Tensor tensor) {
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(tensor);
  auto outputs = model->forward(inputs).toTuple()->elements();
  // Contains boxes and confidences
  std::vector<torch::Tensor> ret;
  for (const auto& item : outputs)
  {
    ret.push_back(item.toTensor());
  }

  return ret;
}

// Convert output tensor to vector of floats
std::vector<float> as_float(torch::Tensor o)
{
  o = o.to(torch::kCPU);
  auto sizes = o.sizes();
  size_t items = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<int>());
  return {o.data<float>(), o.data<float>() + items};
}

// Convert output tensor to vector of long
std::vector<long> as_long(torch::Tensor o)
{
  o = o.to(torch::kCPU);
  auto sizes = o.sizes();
  size_t items = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<int>());
  return {o.data<long>(), o.data<long>() + items};
}

// 1. Read model
std::shared_ptr<torch::jit::script::Module> read_model(std::string model_path) {

  std::shared_ptr<torch::jit::script::Module> model = torch::jit::load(model_path);

  assert(model != nullptr);

  // If we have CUDA
  model->to(at::kCUDA);
  return model;
}

// 2. Forward
std::vector<torch::Tensor> forward(cv::Mat image,
  std::shared_ptr<torch::jit::script::Module> model) {

  // 1. Convert OpenCV matrices to torch Tensor
  torch::Tensor tensor = __convert_image_to_tensor(image);

  // 2. Predict
  std::vector<torch::Tensor> outputs = __predict(model, tensor.to(torch::kCUDA));

  // 3. Convert torch Tensor to vector of vector of floats
  torch::Tensor boxes = torch::squeeze(outputs.at(0));
  torch::Tensor conf = torch::softmax(torch::squeeze(outputs.at(1)), 1);

  return {boxes, conf };
}

// 3. Postprocess
std::tuple<std::string, std::string> postprocess(std::vector<float> probs,
  std::vector<std::string> labels) {

  // 1. Get label and corresponding probability
  auto prob = std::max_element(probs.begin(), probs.end());
  auto label_idx = std::distance(probs.begin(), prob);
  auto label = labels[label_idx];
  float prob_float = *prob;

  return std::make_tuple(label, std::to_string(prob_float));
}
