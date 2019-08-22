## PEP8 Formatting Scripts

`autopep8_file.sh` reformats a single file
`autopep8_all.sh` reformats all .py files in mlod/mlod

#### Usage
```
autopep8_file.sh <file_path.py>
```
```
autopep8_all.sh
```

### PyCharm Setup
Using the external tools, the current file can be passed in the bash script as a parameter.

#### Setup
1. Add an External Tool
- File -> Settings -> Tools -> External Tools
- Add New External Tool (+)
    - Name: Reformat Code
    - Options:
        - [x] Synchronize files after execution
        - [ ] Open console
    - Program: `path to autopep8_file.sh`
    - Parameters: `$FilePath$`
    - Working directory: `$ProjectFileDir$`

2. Add a key binding
- File -> Settings -> Keymap
- External Tools -> External Tools
- Right-click on the new `Reformat Code`-> Add Keyboard Shortcut
    - Ex. `Ctrl + Alt + G`

### Notes
The scripts are set to be read-only so that the formatter does not accidentally reformat them.
