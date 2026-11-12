# upfront-costs

Analyzing and predicting the upfront costs of building electrification. This repo includes a data pipeline for predicting the upfront costs of:

- Air-source heat pumps (ASHPs)
- Ground-source heat pumps (GSHPs)
- Heat pump water heaters (HPWHs)
- Weatherization
- Panel upgrades
- Wiring
- Rooftop solar
- EVs
- EV L2 chargers
- Ranges (stoves)
- Dryers

Other upgrades may be added in the future.

## Navigating this Repository

Pipeline code is stored in the `src/` subdirectory, with a detailed methodology description in `METHODOLOGY.md`. Model evaluation notebooks live in `src/evaluate/`. Testing code lives in `tests/`.

```text
upfront-costs/
├── .github/workflows/           # CI: lint, tests, and pushing init script & DAB to Databricks
├── exploratory/                 # Exploratory analysis of ASHP training data
├── resources/                   # Databricks asset bundle job definition
├── src/
│   ├── 0_process/               # Clean raw data sources
│   ├── 1_generate/              # Generate training features
│   ├── 2_train/                 # Train upfront cost models
│   ├── 3_predict/               # Predict upfront costs for each building in a BuildingSet
│   ├── 4_combine/               # Combine per-upgrade predictions into a single table
│   ├── evaluate/                # Model evaluation notebooks
│   ├── globals.py               # Shared paths and constants
│   ├── util.py                  # General helpers
│   ├── modeling_util.py         # ModelHelper class for training and prediction
│   └── geocoding_util.py        # Geocode addresses to counties
├── tests/                       # Unit tests
├── METHODOLOGY.md               # Upfront cost estimation methodology
├── databricks.yml               # Databricks asset bundle config
├── install-db-requirements.sh   # Cluster init script that installs dependencies
├── requirements-db-15.4.txt     # Pinned requirements for Databricks Runtime 15.4
└── pyproject.toml               # Poetry project and dependencies
```

## Data Sources

The ASHP, GSHP, HPWH, panel upgrade, and wiring upfront cost models use data from the following sources:

- Mass Saves: A heat-pump program in Massachusetts that ran from 2015-2021
- [TechClean CA](https://techcleanca.com/heat-pump-data/heat-pump-data-visuals/): An ongoing program to install heat pumps in California, last pulled in 2024
- [NYSERDA: Field Monitoring Ground Source Heat Pumps on Long Island (2024)](https://www.notion.so/rewiringamerica/GSHP-Field-Data-d1e0b3255a5847e2951227aa0182959f)

Solar costs are predicted using data from [Tracking the Sun](https://emp.lbl.gov/tracking-the-sun) (LBNL).

Weatherization costs are predicted using data from ["The Cost of Decarbonization and Energy Upgrade Retrofit for US Homes"](https://escholarship.org/content/qt0818n68p/qt0818n68p.pdf) (LBNL).

EV costs are predicted using the [PG&E EV Comparison Tool](https://ev.pge.com/compare).

L2 charger costs are predicted using the 2025 projections from the [2019 ICCT EV Charging Cost study](https://theicct.org/wp-content/uploads/2021/06/ICCT_EV_Charging_Cost_20190813.pdf).

Range and dryer costs are predicted using desk research.

For more methodology details, see `METHODOLOGY.md`.

## External users

Running the code in this repo requires access to internal Rewiring America dependencies (e.g., RSMeans factors and [dmlutils](https://github.com/rewiringamerica/dml-utils/)), as well as a Databricks cluster using DBR 15.4.

However, external users may access and use some models attached to specific releases (see "Releases" under "About"). Note that without access to RSMeans geographic adjustment factors, predictions from these models out of the box will be national and not location-specific. Additional details on model usage are included in each associated release.

Bug reports and feature requests can be submitted via Github Issues. For questions or to join our open source community, please reach out to `opensource@rewiringamerica.org`.

## Internal users

### Running this Code

For a guide to setting up local development with python, poetry, and gcloud, see the [dml-utils README](https://github.com/rewiringamerica/dml-utils/blob/main/README.md#one-time-setup).

Use the Upfront costs databricks job to run part or all of the production pipeline.

The pipeline runs in the following order:

1. `src/0_process/`
2. `src/1_generate/`
3. `src/2_train/`
4. `src/3_predict/`
5. `src/4_combine/`

### Prediction on a BuildingSet

The models are currently used to predict the upfront cost of various appliances (or appliance x upgrade combinations, when appliance upfront costs depend on factors determined by the larger upgrade) for each individual building in a given BuildingSet (by default ResStock 2024.2). These tables are saved and used in data sources like the Community Electrification Dashboard (CED) to give the expected upfront costs of appliances and upgrades across the country.

### Contributing

#### Incrementing Versions

Each trained model and saved table is tagged by the release version at the time of creation, distinguishing the artifacts from earlier versions.
If there are any pipeline changes, whether by changing data sources, adding new features, or implementing a new model, the release version in the repository should be incremented.  **Please read [this guide](https://github.com/rewiringamerica/dml-utils/blob/main/README.md#increment-the-version) to versioning before incrementing.** By incrementing, the training pipeline will generate new feature tables and save a new version of each model, all tagged with the new release version number. This will allow us to quickly revert to a previous version of the pipeline if we encounter issues with model/data drift.

#### Testing Changes with the Dev Job

The full pipeline is defined as a Databricks Asset Bundle job in `resources/upfront_costs_job.yml`. To test changes on your branch end-to-end, do the following:

1. Push your branch to GitHub. The dev job runs notebooks from whichever branch you have checked out locally when you deploy.
2. If you have made changes to the project environment and init script (see [Updating Requirements](#updating-requirements)), create a temporary commit that points the job at your branch's init script:
   1. Check out your feature branch in your Databricks workspace Git folder (`Workspace/Repos/<your email>/upfront-costs`), and copy the path to the init script in that folder.
   2. In the YAML file's job cluster definition, change the init script destination to `/Workspace/Repos/<your email>/upfront-costs/install-db-requirements.sh`. Since this is a workspace file rather than a volume, also change the `volumes` key to `workspace`:

      ```yaml
      init_scripts:
        - workspace:
            destination: /Workspace/Repos/<your email>/upfront-costs/install-db-requirements.sh
      ```

3. Deploy your own copy of the job to the dev target with `databricks bundle deploy -t dev`, then run the job. The dev target sets the `mode` parameter to `feature`, so outputs are written to feature tables rather than overwriting the `main` tables.
4. If you made a temporary commit in step 2, revert it with `git revert` before merging.

#### Updating Requirements

Whenever you add or change a requirement to `pyproject.toml` you need to

1. `poetry update` as normal
2. generate requirements files with `dml-gen-requirements` as described in the
   [dml-sample-transmission-line](https://github.com/rewiringamerica/dml-sample-transmission-line?tab=readme-ov-file#updating-dependencies)
   `README.md`, using the 15.4 version.
   This will update a few files, including `install-db-requirements.sh`.

See `dml-utils`' [Adding a dependency](https://github.com/rewiringamerica/dml-utils/blob/main/README.md#adding-a-dependency) for more details.

#### Spell-checker

This repo has [cspell](https://cspell.org/) configured in `cspell.json` for optional (highly recommended) spell-checking. If you're using VSCode, all you need is to install the [cspell extension](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker) and spelling issues will be highlighted. If you're using another IDE, [install cspell](https://cspell.org/docs/installation) then run spell-checking in your command line using `cspell .`. To add new word(s) to the dictionary in VSCode, select some text > Right click > Spelling > Add Words to Dictionary. In other IDEs, words may need to be added manually to `/.cspell/upfront_costs_dict.txt`. For more details, see the [cspell docs](https://cspell.org/docs/getting-started).

#### Naming Notebooks and Tables

Please read [these guidelines](https://www.notion.so/rewiringamerica/Notebook-and-Table-Naming-2c54c9af1a1c8085a478fefcb664d5d7) before naming or renaming notebooks/tables.
