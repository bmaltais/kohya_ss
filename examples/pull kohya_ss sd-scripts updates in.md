## Updating a Local Branch with the Latest sd-scripts Changes

To update your local branch with the most recent changes from kohya/sd-scripts, follow these steps:

1. Add sd-scripts as an alternative remote by executing the following command:

   ```
   git remote add sd-scripts https://github.com/kohya-ss/sd-scripts.git
   ```

2. When you wish to perform an update, execute the following commands:

   ```
   git checkout dev
   git pull sd-scripts main
   ```

   Alternatively, if you want to obtain the latest code, even if it may be unstable:

   ```
   git checkout dev
   git pull sd-scripts dev
   ```

3. If you encounter a conflict with the Readme file, you can resolve it by taking the following steps:

   ```
   git add README.md
   git merge --continue
   ```

   This may open a text editor for a commit message, but you can simply save and close it to proceed. Following these steps should resolve the conflict. If you encounter additional merge conflicts, consider them as valuable learning opportunities for personal growth.