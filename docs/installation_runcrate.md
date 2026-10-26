### Runcrate

[Runcrate](https://www.runcrate.ai/?utm_source=kohya_ss&utm_medium=directory&utm_campaign=gpu_pricing) rents GPU virtual machines by the hour. These steps install kohya_ss on a Runcrate instance with the uv method and open the GUI in your browser through an HTTPS link.

#### Manual installation

1. Create an instance with an NVIDIA GPU and the **Finetune** image (`ubuntu-train`), either from the [deploy form](https://www.runcrate.ai/auth/signup?utm_source=kohya_ss&utm_medium=directory&utm_campaign=gpu_pricing&redirect=%2Fdashboard%2Finstances%3Faction%3Dcreate%26gpu%3DA6000%26template%3Dubuntu-train%26src%3Dkohya_ss) or with the [Runcrate CLI](https://www.runcrate.ai/docs/cli/installation?utm_source=kohya_ss&utm_medium=directory&utm_campaign=gpu_pricing):

   ```shell
   runcrate instances create --name kohya --gpu A6000 --template ubuntu-train --max-runtime 8h
   ```

   Pick the GPU by the model you train; `runcrate instances types --gpu A6000` lists machines and prices. `--max-runtime` stops the instance after 8 hours so it does not keep billing.

2. Open a shell on the instance, from the instance page in the dashboard or with:

   ```shell
   runcrate ssh kohya
   ```

3. Install `uv` and Tk, then clone the repository with its submodules:

   ```shell
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
   apt-get update && apt-get install -y python3-tk
   git clone --recursive https://github.com/bmaltais/kohya_ss.git
   cd kohya_ss
   ```

4. Start the GUI with **`--headless`**, listening on all interfaces. The first start creates the virtual environment and installs the dependencies, which takes a few minutes:

   ```shell
   ./gui-uv.sh --listen 0.0.0.0 --server_port 7860 --headless
   ```

5. In the dashboard, open the instance, add an HTTPS link for port `7860` with **Workspace** access, and click **Open**. Only signed-in members of your workspace can reach it. The GUI is served at an address like `https://7860-<instance-id>.runcrate.app`.

   To reach it without the dashboard instead, give the link **Only these addresses** access with your own IP addresses (add both your IPv4 and IPv6 address if your connection has both), or start the GUI with `--username` and `--password`.

6. Delete the instance when you have downloaded your trained models, to stop billing:

   ```shell
   runcrate instances delete kohya
   ```
