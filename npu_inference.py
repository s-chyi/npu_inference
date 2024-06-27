import torch
from torchvision import models, transforms
from transformers import pipeline, AutoTokenizer, TextStreamer, AutoModelForCausalLM
from intel_npu_acceleration_library import NPUModelForCausalLM
import intel_npu_acceleration_library
from PIL import Image
from pathlib import Path
import openvino as ov
import time

def list_available_devices():
    print("Available CPUs:")
    print("CPU")
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print("\nAvailable GPUs:")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("\nNo GPUs available")

    core = ov.Core()
    print("Openvino available devices:")
    print(core.available_devices)

def test_inference(model, input_data, num_iterations=3):
    
    with torch.no_grad():
        # 预热模型
        _ = model(input_data)
    
    times = []
    for _ in range(num_iterations):
        start_time = time.time()
        with torch.no_grad():
            _ = model(input_data)
        end_time = time.time()
        times.append(end_time - start_time)
        
    for i, t in enumerate(times, 1):
        print(f"Inference {i}: {t*1000:.2f} ms")
    print(f"Average inference time: {sum(times)*1000 / len(times):.2f} ms")

def load_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # 添加额外的批次维度
    
def convert_model(pytorch_model, model_name):
    MODEL_DIRECTORY_PATH = Path("model")
    MODEL_DIRECTORY_PATH.mkdir(exist_ok=True)
    precision = "FP16"
    core = ov.Core()
    model_path = MODEL_DIRECTORY_PATH / "ir_model" / f"{model_name}_{precision.lower()}.xml"

    model = None
    if not model_path.exists():
        model = ov.convert_model(pytorch_model, input=[[1, 3, 224, 224]])
        ov.save_model(model, model_path, compress_to_fp16=(precision == "FP16"))
        print("IR model saved to {}".format(model_path))
    else:
        print("Read IR model from {}".format(model_path))
        model = core.read_model(model_path)

    return core.compile_model(model, "NPU")

def test_mobilenetv2():
    mobilenet_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 测试设备
    list_available_devices()

    # 加载 MobileNetV2 模型
    print("\nTesting MobileNetV2:")
    mobilenet_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).eval()

    ov_compiled_model = convert_model(mobilenet_model, "mobilenetv2")

    image_path = r'C:\Users\User\Desktop\bechmark\data\dog.jpg'  # 替换为本地图像的路径
    input_data_mobilenet = load_image(image_path, mobilenet_transform)

    print("Testing on CPU:")
    cpu_mobilenet_model = mobilenet_model.to('cpu')
    cpu_input_data = input_data_mobilenet.to('cpu')
    test_inference(cpu_mobilenet_model, cpu_input_data)

    if torch.cuda.is_available():
        print("Testing on GPU:")
        mobilenet_model.to('cuda')
        gpu_input_data = input_data_mobilenet.to('cuda')
        test_inference(mobilenet_model, gpu_input_data)

    print("Testing on NPU:")
    # npu_model = intel_npu_acceleration_library.compile(mobilenet_model)
    test_inference(ov_compiled_model, input_data_mobilenet)


def test_llm_inference(model, prefix, streamer, num_iterations=3):
    generation_kwargs = dict(
        input_ids=prefix["input_ids"],
        streamer=streamer,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        max_new_tokens=128,
    )
    _ = model.generate(**generation_kwargs)
    
    times = []
    for _ in range(num_iterations):
        start_time = time.time()
        output = model.generate(**generation_kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
        
    for i, t in enumerate(times, 1):
        print(f"Inference {i}: {t:.6f} seconds")
    print(f"Average inference time: {sum(times) / len(times):.6f} seconds")


def test_phi3():
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    cpu_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", use_cache=True, trust_remote_code=True).eval()
    gpu_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, use_cache=True, trust_remote_code=True).to("cuda").eval()
    npu_model = NPUModelForCausalLM.from_pretrained(model_id, use_cache=True, dtype=torch.float16).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    streamer = TextStreamer(tokenizer, skip_special_tokens=True)
    query = "<|user|>Can you introduce yourself?<|end|><|assistant|>"
    # query = "Can you introduce yourself?"
    prefix = tokenizer(query, return_tensors="pt")

    core = ov.Core()
    model = core.read_model("model.xml")
    npu_model = core.compile_model(model, "NPU")
    print("Model: Phi-3-mini-4k-instruct")

    print("Testing on CPU:")
    test_llm_inference(cpu_model, prefix, streamer)

    print("Testing on GPU:")
    test_llm_inference(gpu_model, prefix.to("cuda"), streamer)
    
    print("Testing on NPU:")
    test_llm_inference(npu_model, prefix, streamer)


if __name__ == "__main__":
    # test_mobilenetv2()
    test_phi3()
