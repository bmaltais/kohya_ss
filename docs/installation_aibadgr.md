### AI Badgr

AI Badgr provides routed GPU capacity for running the `kohya_ss` web GUI from a hosted template. The template exposes the GUI on port `7860` and supports a persistent endpoint, cost cap, health check, auto teardown, logs, and receipts.

#### Pre-built AI Badgr template

To run from the pre-built AI Badgr template, you can:

1. Open the AI Badgr template by clicking on <https://aibadgr.com/gpu/launch?template=kohya-ss>.

2. Sign in or create a free account.

3. Configure the workload with your desired GPU and maximum cost cap.

4. Launch the template from the web UI.

5. Once deployed, open the returned endpoint URL to access the `kohya_ss` GUI on port `7860`.

#### Workload details

The AI Badgr `kohya-ss` template is configured for:

- Docker image: `bmaltais/kohya-ss-gui:latest`
- GPU: RTX 4090 with 8+ GB VRAM
- GPU count: `1`
- Exposed port: `7860`

After the endpoint starts, upload datasets and configure training runs from the browser.

#### AI Badgr CLI

If you use the AI Badgr CLI, install it and authenticate once:

```shell
npm install -g badgr-cli
badgr login
```

Then serve the `kohya-ss` template with a maximum cost cap:

```shell
badgr serve template kohya-ss --max-cost 5
```

AI Badgr returns an endpoint URL. Stop billing any time with:

```shell
badgr down <id>
```

#### README badge

To add a launch badge to your own README, use:

```markdown
[![Run on AI Badgr](https://aibadgr.com/badge.svg)](https://aibadgr.com/gpu/launch?template=kohya-ss)
```

For provider lists, use the plain link:

```markdown
[AI Badgr](https://aibadgr.com/gpu/launch?template=kohya-ss)
```
