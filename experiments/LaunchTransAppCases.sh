#!/bin/bash

# Array declarations (fixed syntax)
all_case=("cooker_case" "dishwasher_case" "desktopcomputer_case" "ecs_case" "pluginheater_case" "tumbledryer_case" "tv_greater21inch_case" "tv_less21inch_case" "laptopcomputer_case")
models=("TransApp" "TransAppPT")
# dim_model = ("96")
frac=("1")

for case in "${all_case[@]}"; do
  for model in "${models[@]}"; do
      for f in "${frac[@]}"; do
        bash RunTransAppClassif.sh "$case" "$model" 96 "$f"
      done
    done
  done
done
