# Nvidia-MTC
Compile fast ML models for NVIDIA GPUs.

## Attribution
- [TVM](https://tvm.apache.org/) - ML powered ML compiler.
- [TCLPack](https://tlcpack.ai/) - Prebuilt TVM binaries for NVIDIA CUDA
- [Modal](https://modal.com/) - Serverless GPU FAAS platform.

## Prerequisites
You will need a [Modal](https://modal.com/) account. The app should fit within free tier.

1. Create a `.env` file in project root with the following contents:
```
APP_NAME=some-random-app-name
MODAL_WORKSPACE=your-modal-workspace
```
