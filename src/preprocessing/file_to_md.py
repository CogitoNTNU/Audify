from markitdown import MarkItDown
import os
from urllib.parse import urlparse

#A function that converts other file types to Markdown files
def file_to_md(input_file: str, output_filename: str):

    md = MarkItDown()
    parsed = urlparse(input_file)
    #To get a relative file path (so we can place it correctly)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Handle URLs
    if parsed.scheme in ("http", "https"):
        md_content = md.convert(input_file)
    else: 
        # Handle local files
        # If input_file is already an absolute path, use it; otherwise join with script_dir
        if os.path.isabs(input_file):
            file_path = input_file
        else:
            file_path = os.path.abspath(os.path.join(script_dir, input_file))

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fant ikke filen: {input_file}")
        
        md_content = md.convert(file_path)
    
    # Write the markdown content to file (for both URLs and local files)
    base_dir = os.path.join(script_dir, "..", "..", "data", "markdown")
    os.makedirs(base_dir, exist_ok=True)
    output_path = os.path.join(base_dir, output_filename)

    #Writes/creates a new file that is saved in the 'markdown' folder
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content.text_content)

# file_to_md("https://www.theguardian.com/uk-news/2025/nov/02/andrew-to-be-stripped-of-naval-title-says-uk-defence-secretary","test.md")
