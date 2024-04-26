def freezeall(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def analysis_parameter(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e9:.2f}B")
    print(f"Trainable parameters: {trainable_params}")
    # print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    print('loaded model lora')
    print('8'*8)