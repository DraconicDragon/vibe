from rope.base.project import Project
from rope.refactor.rename import Rename

# Path to your project root (VERY important)
project = Project(".")

# Path to the folder/module you want to rename
resource = project.get_resource("autotagger")

# Create rename refactoring
renamer = Rename(project, resource)

# Perform rename
changes = renamer.get_changes("vibe")

# Apply changes
project.do(changes)

# Close project
project.close()