from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def build_text_chunks(records):
    """
    Break each record's context text into overlapping chunks.
    Overlap keeps some shared context between consecutive chunks
    so meaning isn't lost at the boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunk_list = []
    for record in records:
        text_chunks = splitter.split_text(record["context"])
        chunk_list += text_chunks

    return chunk_list


def main():
    from data_loader import get_data

    dataset = get_data()
    chunk_list = build_text_chunks(dataset)

    print(f"\nNumber of chunks created: {len(chunk_list)}")
    if chunk_list:
        print(f"Example chunk:\n{chunk_list[0]}")


if __name__ == "__main__":
    main()
