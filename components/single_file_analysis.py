import streamlit as st
import os
import re
import yake
from yake.highlight import TextHighlighter



def single_file_analysis(file_path: str):
    st.header("Single File Analysis")
    # Placeholder for file selector
    # st.title("Text File Viewer")
    
    def parse_filename(filename):
        # Extract date and segment from filename like "2024-12-30-segment-06.txt"
        try:
            date_part = filename.split('-segment-')[0]
            segment = int(filename.split('-segment-')[1].replace('.txt', ''))
            year, month, day = map(int, date_part.split('-'))
            return (year, month, day, segment)
        except:
            return (0, 0, 0, 0)  # Return zeros for invalid filenames

    if file_path and os.path.isdir(file_path):
        text_files = [f for f in os.listdir(file_path) if f.endswith('.txt')]
        text_files.sort(key=parse_filename)
        if text_files:
            selected_file = st.selectbox(
                "Select a file to view",
                text_files,
                key="file_selector"
            )
            content = ""
            try:
                with open(os.path.join(file_path, selected_file), 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                # Updated pattern to split by punctuation followed by space and specific characters
                split_pattern = r"(?<=[.!?])\s(?=\[)|(?<=[.!?])\s(?=[A-Z]+:)|(?<=[.!?])\s(?=\()"
                
                # Split content into lines and format
                formatted_lines = []
                current_line = ""
                
                for line in content.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Split the line based on the pattern
                    parts = re.split(split_pattern, line)
                    
                    for part in parts:
                        part = part.strip()
                        if part:
                            if current_line:
                                formatted_lines.append(current_line)
                                formatted_lines.append('')  # Add empty line
                            current_line = part
                
                # Add the last line if not empty
                if current_line:
                    formatted_lines.append(current_line)
                
                formatted_content = '\n'.join(formatted_lines)
                
                # Create a container with scrollable text area
                with st.container():
                    st.markdown("### File Contents")
                    st.text_area(
                        "File Contents",  # Non-empty label
                        value=formatted_content,
                        height=500,
                        key="file_contents",
                        label_visibility="collapsed"  # Hide the label
                    )
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        else:
            st.warning("No text files found in the specified directory.")
    else:
        st.warning("Please enter a valid folder path containing text files.")


    # Placeholder for NER
    st.subheader("Named Entity Recognition (NER)")
    st.write("Highlighting key entities... (Not Implemented)")
    # raise NotImplementedError("NER function needs implementation.")
    if formatted_content:
        kw_extractor = yake.KeywordExtractor(dedupLim = 0.7)
        keywords = kw_extractor.extract_keywords(formatted_content)
        keywords = [kw[0] for kw in keywords if 'CLIP' not in kw[0] and 'CNN' not in kw[0] and 'VIDEO' not in kw[0]]
        
        st.write(keywords)

        th = TextHighlighter(max_ngram_size = 3, highlight_pre = ":blue-background[", highlight_post= "]")
        highlighted_text = th.highlight(formatted_content, keywords)
        st.markdown(highlighted_text)
