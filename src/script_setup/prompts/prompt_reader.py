def load_prompt_md(path: str):
    if not path.endswith('md'):
        return None
    else:
        markdown_string = ''
        try:
            with open(path, "r", encoding="utf-8") as file:
                markdown_string = file.read()
        except Exception:
            markdown_string = ''
        return markdown_string
            
            