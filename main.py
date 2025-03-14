import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

def parse_ctm(ctm_file):
    """Parse the reference CTM file and store word occurrences including confidence scores."""
    index = []
    with open(ctm_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue  # Skip malformed lines
            file_id, channel, start_time, duration, word, confidence = parts[:6]
            index.append({
                'file_id': file_id,
                'channel': channel,
                'start_time': float(start_time),
                'duration': float(duration),
                'word': word,
                'confidence': float(confidence)
            })

    print("Done parsing")
    return pd.DataFrame(index)

def parse_queries(query_file):
    """Parse the XML file containing KWS queries including multi-word phrases."""
    tree = ET.parse(query_file)
    root = tree.getroot()
    queries = []
    for kw in root.findall('kw'):
        kwid = kw.get('kwid')
        phrase = kw.find('kwtext').text.strip()
        queries.append({'kwid': kwid, 'phrase': phrase, 'words': phrase.split()})
    print("Done parsing queries")
    return pd.DataFrame(queries)

def perform_kws_search(index_df, queries_df):
    """Search index for keyword queries, handling multi-word phrases."""
    hits = []
    
    for _, query in tqdm(queries_df.iterrows(), total=len(queries_df)):
        words = query['words']
        num_words = len(words)
        
        # Look for possible phrase occurrences in the index
        for i in range(len(index_df) - num_words + 1):
            segment = index_df.iloc[i : i + num_words]
            
            if list(segment['word']) == words:
                # Ensure time constraints
                time_diffs = segment['start_time'].diff().fillna(0)
                if all(time_diffs[1:] <= 0.5):
                    start_time = segment.iloc[0]['start_time']
                    end_time = segment.iloc[-1]['start_time'] + segment.iloc[-1]['duration']
                    duration = end_time - start_time
                    confidence = segment['confidence'].mean()  # Average confidence over phrase
                    
                    hits.append({
                        'kwid': query['kwid'],
                        'file_id': segment.iloc[0]['file_id'],
                        'channel': segment.iloc[0]['channel'],
                        'start_time': start_time,
                        'duration': duration,
                        'score': confidence,  # Use mean confidence
                        'decision': "YES"
                    })
    
    print("Done performign kws search")
    return pd.DataFrame(hits)

def save_hits_to_ctm(hits_df, output_file):
    """Save the detected KWS hits in CTM format."""
    with open(output_file, 'w') as f:
        for _, hit in hits_df.iterrows():
            f.write(f"{hit['file_id']} {hit['channel']} {hit['start_time']:.2f} {hit['duration']:.2f} {hit['kwid']} {hit['score']:.6f}\n")

def save_hits_to_xml(hits_df, output_file, kwlist_filename="IARPA-babel202b-v1.0d_conv-dev.kwlist.xml", language="swahili"):
    """Save the detected KWS hits in XML format following the expected structure."""
    root = ET.Element("kwslist", kwlist_filename=kwlist_filename, language=language, system_id="")
    
    grouped_hits = hits_df.groupby("kwid")
    for kwid, group in grouped_hits:
        detected_kwlist = ET.SubElement(root, "detected_kwlist", kwid=kwid, oov_count="0", search_time="0.0")
        for _, hit in group.iterrows():
            kw = ET.SubElement(detected_kwlist, "kw", file=hit['file_id'], channel=str(hit['channel']))
            ET.SubElement(kw, "tbeg").text = str(hit['start_time'])
            ET.SubElement(kw, "dur").text = str(hit['duration'])
            ET.SubElement(kw, "score").text = str(hit['score'])  # Use confidence score
            ET.SubElement(kw, "decision").text = hit['decision']
    
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

def main():
    ctm_file = "lib/ctms/reference.ctm"
    query_file = "lib/kws/queries.xml"
    output_ctm_file = "output/reference.ctm"
    output_xml_file = "output/reference.xml"
    
    index_df = parse_ctm(ctm_file)
    queries_df = parse_queries(query_file)
    hits_df = perform_kws_search(index_df, queries_df)
    save_hits_to_ctm(hits_df, output_ctm_file)
    save_hits_to_xml(hits_df, output_xml_file)
    print(f"KWS search completed. Outputs saved to {output_ctm_file} and {output_xml_file}")
    
if __name__ == "__main__":
    main()
