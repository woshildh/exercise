import sys,os

program="python"
arguments=["./called_process.py"]
os.execvp(program,(program,)+tuple(arguments))
print("Good bye") #这行永远不会执行
