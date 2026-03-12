from selfx.backend.features import AnalysisManager

manager = AnalysisManager("1h")
missing = manager.get_non_analyzed_intervals("2023-08-14", "2023-08-15")
for interval in missing:
    print(interval)
print("Finished")