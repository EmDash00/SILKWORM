# SILKWORM

## What is SILKWORM?
**S**IL  
**I**ntegrated  
**L**ibraries and  
**K**nowledgebase for  
**W**ork,  
**O**ngoing  
**R**esearch, and  
**M**anagement
These libraries were made for use by SIL at IUB for our projects; however, we have elected to make these following general utilities open source.

## What's in each branch?
| Branch			 | Description															                                         |
| ------------ | --------------------------------------------------------------------------------- |
| master       | Tested, fully working, fully integrated, and reviewed by 2 people.                |
| develop		   | Tested, fully working, not integrated. Reviewed by comitters.                     | 
| silktime     | silktime package                                                                  |

Branches that aren't master or develop are dedicated to packages. Large packages may have multiple branches and follow the convention `package/feature`

## Currently Available Packages
* [silktime](silkworm/silktime)
	Standard utilties for high fidelity timing.

## How Do I Use SILKWORM in My Project?

~~For noncontributors simply looking for the packages, do `pip3 install silkworm`~~
Currently there is no working PyPI package.

### Adding silkworm to your project
Contributors also have the option adding silkworm as a gitmodule to their project so that they don't have to wait for a rebuild to PyPI for approval of new features that they need in their project.
- Clone SILKWORM and dump the silkworm package into your project.
```shell
git clone https://github.iu.edu/emchow/SILKWORM.git
cp -R SILKWORM/silkworm path/to/project/silkworm/folder
```

- Add SILKWORM as a git submodule. **(Recommended)**

```shell
cd /path/to/project/folder
git submodule add https://github.iu.edu/emchow/SILKWORM.git 
git commit -m "Added SILKWORM to my project"
```

### Managing SILKWORM in a project
Adding SILKWORM as a submodule allows you to recieve updates without recloning. There are two ways to recieve updates: 
	
- You may use `git fetch` within the subdirectory your submodule resides in
- Use `git submodule update --remote` somewhere in your git project's directory (not inside of a submodule). This will update all submodules. **(Recommended)**



To clone such projects, and their submodules use `git clone --recurse-submodules` 

It is recommended to keep all branches with a copy of SILKWORM if you do this. Not doing so will cause issues when switching branches.

You can read more about git submodules [here](https://git-scm.com/book/en/v2/Git-Tools-Submodules)

**Note to Contributors**: I do not recommend you edit SILKWORM from the submodule unless absolutely necessary. Merge conflicts can be difficult to resolve in your project
if they occur. Also there is a high chance of causing future merge conflicts from whatever package you edited to develop.


