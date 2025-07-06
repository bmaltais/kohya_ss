### Runpod

#### Manual installation

To install the necessary components for Runpod and run kohya_ss, follow these steps:

1. Select the Runpod pytorch 2.2.0 template. This is important. Other templates may not work.

2. SSH into the Runpod.

3. Clone the repository by running the following command:

   ```shell
   cd /workspace
   git clone --recursive https://github.com/bmaltais/kohya_ss.git
   ```

4. Run the setup script:

   ```shell
   cd kohya_ss
   ./setup-runpod.sh
   ```

5. Run the GUI with:

   ```shell
   ./gui.sh --share --headless
   ```

   or with this if you expose 7860 directly via the runpod configuration:

   ```shell
   ./gui.sh --listen=0.0.0.0 --headless
   ```

6. Connect to the public URL displayed after the installation process is completed.

#### Pre-built Runpod templates

To run from a pre-built Runpod template, you can:

1. Open the Runpod template by clicking on one of the template links in the table below.

2. Deploy the template on the desired host.

3. Once deployed, connect to the Runpod on HTTP 3010 to access the kohya_ss GUI. You can also connect to auto1111 on HTTP 3000.

| Runpod Template Version                                                                 | Runpod Template Description                        |
|-----------------------------------------------------------------------------------------|----------------------------------------------------|
| [CUDA 12.4 template](https://runpod.io/console/deploy?template=uajca40f1z&ref=2xxro4sy) | Template with CUDA 12.4 for non-RTX 5090 GPU types |
| [CUDA 12.8 template](https://runpod.io/console/deploy?template=8y5a02q55r&ref=2xxro4sy) | Template with CUDA 12.8 for RTX 5090 GPU type      |

