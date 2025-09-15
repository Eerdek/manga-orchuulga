from translate_lingva import translate_lines_impl

if __name__ == "__main__":
    sample_lines = [
        "Hello world!",
        "This is a test.",
        "Goodbye!"
    ]
    # Run with local_only=True to test fallback and TM
    results = translate_lines_impl(sample_lines, source_lang="en", target_lang="mn", local_only=True)
    for src, tgt in zip(sample_lines, results):
        print(f"{src} -> {tgt}")
