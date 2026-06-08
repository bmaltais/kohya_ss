### AI Badgr

AI Badgr provides a hosted GPU template for launching `kohya_ss` without a manual server setup.

#### Pre-built AI Badgr template

To run from the pre-built AI Badgr template, you can:

1. Open the AI Badgr template by clicking on <https://aibadgr.com/gpu/launch?template=kohya-ss>.

2. Launch the template with your desired GPU and cost settings.

3. Once deployed, connect to the service endpoint shown by AI Badgr to access the `kohya_ss` GUI.

#### AI Badgr CLI

If you use the AI Badgr CLI, you can serve the `kohya-ss` template with a maximum cost cap:

```shell
badgr serve template kohya-ss --max-cost 5
```
