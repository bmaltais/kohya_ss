# Pull sd-scripts update in a local branch

For reference for anyone that would like to pull the latest from kohya/sd-scripts, you can:

`git remote add sd-scripts https://github.com/kohya-ss/sd-scripts.git`

to add it as an alternative remote, then when you want to update:

```
git checkout dev
git pull sd-scripts main
```

or, if you want the absolute latest and potentially broken code:

```
git checkout dev
git pull sd-scripts dev
```

You'll probably get a conflict for the Readme, but you can get around it with:

```
git add README.md
git merge --continue
```

which will probably open a text editor for a commit message, but you can just save and close that and you should be good to go. If there are more merge conflicts than that, you now have a potential learning experience and chance for personal growth.