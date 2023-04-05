# Contributing

- [Adding a new argument](#adding-a-new-command-line-argument-to-the-setup-workflow)

## Adding a New Command Line Argument to the Setup Workflow

<details>
<summary>1. Modify <code>install_config.yml</code></summary>

- Add the new argument under the appropriate section (`setup_arguments` or `gui_arguments`) with a name, description, and default value. For example:
    
   ```yaml
    new_argument:
      description: "Description of the new argument"
      default: "default_value"
   ```
</details>

<details>
<summary>2. Modify <code>setup.sh</code></summary>

- Add a new case in the `getopts` loop to handle the new argument, including the short and long options.

```bash
    n | new-argument) CLI_ARGUMENTS["NewArgument"]="$OPTARG" ;;
   ```

- Add a line to set the default value if it's not in the config file.

```bash
    config_NewArgument="${config_NewArgument:-default_value}"
   ```
  
- Add a line to override the config value with CLI arguments.

```bash
    NEW_ARGUMENT="$config_NewArgument"
   ```
</details>

<details>
<summary>3. Modify <code>setup.ps1</code></summary>

- Add a new parameter to the `param` block of the `Get-Parameters` function.

```powershell
    [string]$NewArgument = ""
   ```
- Add a new entry to the `$Defaults` hashtable for the new argument.

```powershell
    'NewArgument' = 'default_value'
   ```
</details>

<details>
<summary>4. Modify <code>launcher.py</code></summary>

- Add a new argument to the `argparse.ArgumentParser` instance.

```python
    parser.add_argument('--new-argument', default=None, help='Description of the new argument')

   ```
- Add a line to load the value from the config file or use the default value.

```python
    new_argument = config.get('new_argument', 'default_value')
   ```

- Add a line to override the config value with CLI arguments if provided.

```python
    new_argument = args.new_argument if args.new_argument else new_argument
   ```
</details>

<details>
<summary>5. Modify <code>setup.bat</code></summary>

- Locate the section where default values for command-line options are set, and add a default value for the new option. For example, if the new option is `--new-option`, add a line like `set NewOption=default_value`.

```batch
    set NewOption=default_value
  ```

- Locate the section that parses command-line arguments (starting with `:arg_loop` and ending with `goto arg_loop`). Add a new conditional statement to handle the new option. For example, if the new option is `--new-option`, add a line like `if /i "%~1"=="--new-option" (shift & set NewOption=%1) & shift & goto arg_loop`.

```batch
    if /i "%~1"=="--new-option" (shift & set NewOption=%1) & shift & goto arg_loop
   ```

- Update the `Args` variable to include the new option. For example, if the new option is `--new-option`, add `--new-option "%NewOption%"` to the `Args` variable definition.

```batch
    --new-option "%NewOption%"
   ```

    - Update the `launcher.py` call to pass the new option as an argument. The `Args` variable already includes the new option, so this step is not necessary.
</details>

After these steps, the new argument will be properly handled in the installation process in all files on all operating systems.