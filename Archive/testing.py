import os
import platform

def showFileExplorer(file):  # Path to file (string)
    if platform.system() == "Windows":
        import os
        os.startfile(file)
    elif platform.system() == "Darwin":
        import subprocess
        subprocess.call(["open", "-R", file])
    else:
        import subprocess
        subprocess.Popen(["xdg-open", file])

