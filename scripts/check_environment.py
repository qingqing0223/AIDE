import platform,sys,torch
print("python:",sys.version); print("platform:",platform.platform()); print("torch:",torch.__version__)
print("cuda_available:",torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda:",torch.version.cuda)
    for i in range(torch.cuda.device_count()): print(i,torch.cuda.get_device_name(i))
