# French Sentence Boundary Detection (SBD) Model Building

I created this model with the hopes of restoring sentence boundaries/punctuation from Youtube's auto-generated subtitles. This would be used for further NLP analysis to determine a corpus' 'difficulty'. That repo is located [here](https://github.com/cofinley/content-difficulty-metrics).

This is essentially the huggingface [token classification example](https://github.com/huggingface/transformers/tree/master/examples/pytorch/token-classification) (`run_ner.py`) with some preprocessing on my part.

I used a dump of the [French opensubtitles corpus](https://opus.nlpl.eu/OpenSubtitles-v2018.php) which gave me real, punctuated sentences (one on each line). The resulting file is about 3.5 GB.

With that, I tagged the start of the sentence using 'B-SENT', removed all punctuation and capitalization, then collected then sentences into bodies of text (batches of 64 sentences) to simulate a large, un-punctuated body of text one might encounter with Youtube auto-generated subtitles. This can all be seen in `preprocessing.py`.

The final model was uploaded to the huggingface model hub [here](https://huggingface.co/cfinley/punct_restore_fr).
