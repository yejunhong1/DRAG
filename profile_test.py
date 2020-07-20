import cProfile
from main import *
import pstats

print("Begin profiling...")
cProfile.run("main()", filename="profile_result.out", sort="cumulative")
print("Finish profiling...")

p = pstats.Stats("profile_result.out")
p.strip_dirs().sort_stats("cumulative", "name").print_stats(0.5)

