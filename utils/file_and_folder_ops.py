def remove_path_after_folder(path, folder):
    # Split the path into a list of directories and files
    dirs = path.split('/')

    # Find the index of the specified folder in the list
    folder_index = dirs.index(folder)

    # Create a new list with the directories up to the specified folder
    new_dirs = dirs[:folder_index + 1]

    # Join the directories in the new list to create the modified path
    new_path = '/'.join(new_dirs)

    return new_path
