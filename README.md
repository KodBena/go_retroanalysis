# go_retroanalysis
Retroanalysis for go. Find contexts that make a certain possibly unusual move the right choice in that situation.

This is a proof of concept piece. It works well enough, but if you want to use it then you need to provide your own KataGo driver (my local files use a KataGoClient class and a KataGoServer UNIX socket backend -- they are based on Lightvector's https://github.com/lightvector/KataGo/blob/master/python/query_analysis_engine_example.py which, as I understand it, is MIT-licensed) as I'm not a license lawyer and therefore unable to pull the right pieces together in a way that harmonizes with license obligations. I didn't want to bother Lightvector with questions of this sort.

To use it run mkdir output; python -i ...
>>> v = start_test(random_drop=True, m=20000);
The folder output will be populated by some SGF files (best_... for best-til-now seen nodes, solution_... for nodes that make the desired move proper)
