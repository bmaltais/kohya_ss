## Updating a Local Submodule with the Latest sd-scripts Changes

To update your local branch with the most recent changes from kohya/sd-scripts, follow these steps:

1. When you wish to perform an update of the dev branch, execute the following commands:

   ```bash
   cd sd-scripts
   git fetch
   git checkout dev
   git pull origin dev
   cd ..
   git add sd-scripts
   git commit -m "Update sd-scripts submodule to the latest on dev"
   ```

   Alternatively, if you want to obtain the latest code from main:

   ```bash
   cd sd-scripts
   git fetch
   git checkout main
   git pull origin main
   cd ..
   git add sd-scripts
   git commit -m "Update sd-scripts submodule to the latest on main"
   ```
