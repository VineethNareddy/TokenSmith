import sys
sys.path.insert(0, ".")
import pathlib
import statistics

from src.config import RAGConfig
from src.preprocessing.chunking import DocumentChunker
from src.preprocessing.extraction import extract_sections_from_markdown

def main():

    data_dir = pathlib.Path("data")
    md_files = sorted(data_dir.glob("*.md"))
    if not md_files:
        print("ERROR: No markdown files found in data/")
        sys.exit(1)
    
    target_md = md_files[0]

    sections = extract_sections_from_markdown(str(target_md), exclusion_keywords=['questions', 'exercises', 'summary', 'references'])

    base_config = RAGConfig()
    
    modes = [
        "recursive_sections",
        "paragraph",
        "fixed_size",
        "context_aware",
        "hybrid_paragraph_fixed",
        "hybrid_paragraph_context_aware"
    ]

    results = []

    for mode in modes:
        base_config.chunk_mode = mode
        base_config.chunk_config = base_config.get_chunk_config()
        
        chunker = DocumentChunker(strategy=base_config.get_chunk_strategy(), keep_tables=True)
        
        all_chunks = []
        for c in sections:
            sub_chunks = chunker.chunk(c['content'])
            all_chunks.extend(sub_chunks)

        if not all_chunks:
            print(f"Note: {mode} produced 0 chunks.")
            continue

        # Calculate metrics
        char_lengths = [len(c) for c in all_chunks]
        word_lengths = [len(c.split()) for c in all_chunks]

        stats = {
            "strategy": mode,
            "total_chunks": len(all_chunks),
            "avg_chars": sum(char_lengths) / len(char_lengths),
            "avg_words": sum(word_lengths) / len(word_lengths),
            "max_chars": max(char_lengths),
            "min_chars": min(char_lengths),
        }
        
        if len(char_lengths) > 1:
            stats["std_dev_chars"] = statistics.stdev(char_lengths)
        else:
            stats["std_dev_chars"] = 0.0

        results.append(stats)

    csv_file = "scripts/chunk-stats/chunk_stats.csv"
    with open(csv_file, "w") as f:
        f.write("strategy,total_chunks,avg_chars,avg_words,max_chars,min_chars,std_dev_chars\n")
        for r in results:
            f.write(f"{r['strategy']},{r['total_chunks']},{r['avg_chars']:.2f},{r['avg_words']:.2f},{r['max_chars']},{r['min_chars']},{r['std_dev_chars']:.2f}\n")
    
    print(f"\nAnalysis complete. Please look at statistics saved to {csv_file}")

if __name__ == "__main__":
    main()
