import os

def remove_comments_from_latex(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.tex'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            with open(file_path, 'w') as file:
                for line in lines:
                    if not line.strip().startswith('%'):
                        file.write(line)

# Usage
folder_path = '/Users/jiarui/Downloads/yktcvvcxtsnmdtmyqbjqxkqtmpxrfqpt'
remove_comments_from_latex(folder_path)
