# Rewiring America Upfront Cost Estimation Methodology

The methodology for Rewiring America’s upfront cost estimates is slightly different for each type of electrification upgrade. These upgrades can be grouped into two categories: those we assume have *location-varying* costs (for example, a heat pump installation in San Francisco, CA is likely to be more expensive than a heat pump installation in Montgomery, AL), and those we assume have *location-invariant* costs (e.g., we assume an induction stove costs the same no matter where you are).

## Location-Varying Upfront Costs

Upgrades we assume to have location-varying upfront costs include air-source and ground-source heat pumps, heat pump water heaters, panels & wiring, insulation, and solar.

For these location-varying upfront costs, we first develop national cost estimates, then multiply by [RSMeans](https://www.rsmeansonline.com/) cost factors to convert from national costs to county-specific costs. The upfront cost estimates for these projects therefore account for how the local cost of electrification differs from the national average.

### Air-source Heat Pumps (ASHPs)

#### Model Training

We use three large ASHP installation datasets to estimate ASHP upfront costs:

- [Massachusetts Residential Air-Source Heat Pump Program](https://www.masscec.com/public-records-requests) (scroll to “Clean Heating and Cooling Project Databases”) | 21,000 projects, including 3,500 whole-home heat pumps | 2014-2019
- [Massachusetts Whole Home Pilot](https://www.masscec.com/public-records-requests) (scroll to “Clean Heating and Cooling Project Databases”) | 158 projects, all whole-home heat pumps | 2019-2021
- [TECH Clean California](https://techcleanca.com/heat-pump-data/heat-pump-data-visuals/) | 9,000 projects, all whole-home heat pumps | 2021-2022 (more updated data is available, but our model currently only incorporates data through 2022)

For each project in the dataset, we adjust the cost to represent a present-day national average by correcting for inflation using a [Construction Price Index](https://fred.stlouisfed.org/series/PPIDCS) (since installations took place over the past decade) and correcting for location-specific materials & labor costs using RSMeans cost factors (since Massachusetts and California are relatively expensive markets). Based off this corrected data, we develop a quantile regression model that predicts ASHP cost given six pieces of information about a home. We used a quantile rather than a linear regression to capture the median project cost rather than the mean project cost, since the project cost distribution has a long upper tail.

1. Square footage
2. Heat pump efficiency (as measured by HSPF)
3. Heat pump size in tons
4. Whether the installation was a cold climate (note that this may encapsulate both the installation of a cold-climate heat pump, and other complexities related to installing heat pumps in cold climates, such as requiring more precise load calculations)
5. Whether the ductwork needs to be replaced or upgraded
6. Whether a panel upgrade is required

#### Prediction

Not all six features listed above are always available, so imputation is necessary:

- Heat pump size: to determine the required ASHP size in tons for a given home square footage and climate zone, we follow [ACCA Manual S](https://view.acca.org/CUp2kG). ACCA Manual S sizes ASHPs primarily for cooling, so we calculate the required tons of cooling capacity per square footage for a given climate zone based on common HVAC rules of thumb from [HVAC Direct](https://hvacdirect.com/sizing-air-conditioner-and-heater.html). Per ACCA Manual S, we add a one-ton oversizing allowance in cold and dry climates. In cold climates that are not dry, we apply an oversizing allowance to the cooling tonnage dictated by the type of ASHP (15% oversized for the minimum-efficiency single-stage heat pump, 30% oversized for the medium- or maximum-efficiency variable-speed heat pumps).
- Panel/ductwork upgrades: whether an installation necessitated a panel upgrade or ductwork is not visible in the Mass Residential ASHP Program data. In the absence of a prior for these probabilities we impute `False` for these variables.

We then use the model to predict upfront costs for a given climate zone, square footage range, heat pump size, and heat pump efficiency. Finally, these upfront costs are tailored to a specific county using RSMeans cost factors.

### Ground-source Heat Pumps (GSHPs)

Unfortunately, available datasets on the upfront cost of GSHPs are limited. The GSHP upfront cost model is trained on data from two sources:

1. The Massachusetts [Ground-source Heat Pump Program](https://www.masscec.com/public-records-requests) (scroll to “Clean Heating and Cooling Project Databases”)
2. [NYSERDA GSHP Field Data](https://app.notion.com/p/d1e0b3255a5847e2951227aa0182959f?pvs=25)

Data from these sources are put together in a csv file with a data source tag for future developments or addition of future datasets from different resources.

Preprocessing steps applied to MassCEC data:

- Building Size (Square Feet) ⇒ remove all buildings w/o reported building size
- Pump type ⇒ remove water to water pump type
- Loop Config ⇒ remove all config except Closed loop - vertical

Exploring this data, here are the main takeaways:

- Pair plot shows a positive linear relationship between total system cost and features like building size and total system capacity.
- "building sq ft" and "total capacity tons" are selected as independent variables (i.e., features) to predict "total system cost" as our dependent variable (i.e., target).
- Total bore depth is highly correlated with total system cost; however, it is also highly correlated with building sq ft. For this reason we generate a total bore depth feature but do not use it to train the model.
- We experimented with numerous model architectures (LR, RF, GB Trees, quantile regression, CatBoost, XGBoost). The top performing architecture chosen based on RMSE and R-squared was XGBoost.

See [GSHP implementation and pipeline evaluations](https://app.notion.com/p/rewiringamerica/GSHP-implementation-and-transmission-line-evaluations-5e93e5a2e7894f33a7ecb329df970fdf) (internal) for more context.

As with the ASHP model, for each project in the dataset, we correct for location-specific materials & labor costs using RSMeans cost factors. **Unlike the ASHP model, however, the GSHP data is not inflation-adjusted.** This decision is based on investigation in early 2025 which concluded that the margin drop on GSHP is much higher than inflation. In other words, GSHPs are the earliest on the adoption curve so the price of a GSHP is expected to drop (due to technological development) faster than inflation.

Then an XGBoost regressor is trained to predict the adjusted cost using two features: square footage and total capacity of the GSHP.

This model is used to predict upfront costs for all single-family occupied homes in ResStock associated with a GSHP upgrade.

### Heat Pump Water Heaters (HPWHs)

We use the [TECH Clean California](https://techcleanca.com/public-data/maps-and-graphs/) dataset, which has data from over 1,200 heat pump water heater installations, to estimate HPWH costs. Similar to the ASHP data, we adjust the cost of each project to represent a present-day national average by correcting for inflation and location-specific materials & labor costs. The main predictor of HPWH cost is whether the household switches from a fossil fuel water heater, which likely requires electrical work, or from an electric resistance water heater. We therefore separately calculate the median and quartile costs for households switching from fossil and for households switching from electric resistance.

We subtract the circuit cost (see Panels & Wiring below) from the median and quartile costs for switching from a fossil fuel water heater to represent the cost of the water heater alone, without the cost of the circuit. We use this approach rather than directly reporting the cost of electric resistance replacement because the TECH Clean California dataset has many more datapoints from households switching from fossil fuel water heaters, so the distribution more accurately captures the range of costs that a household may experience when installing a HPWH.

### Wiring

Wiring upfront costs are estimated as the difference in median upfront cost between TECH Clean CA projects replacing an electric resistance water
heater and projects replacing a gas water heater. This assumes that every gas water heater replacement requires a new branch circuit (wiring), and every ER water heater replacement does not, and the only systematic difference in upfront cost between the two is attributable to the branch circuit.

### Panel Upgrades

Panel upgrade costs vary by target service amperage where cost is calculated using LBNL absolute level multiplied by RSMeans-derived amperage multiplier. [LBNL/NREL NHPC 2024 deck](https://homes.lbl.gov/sites/default/files/2024-04/BPA_NHPC24-Affordable%20Electrification_final.pdf) reports a median main-service-panel upgrade of $3,499, range $1k-$5k (2023, national). A 200A upgrade is assigned this median upfront cost.

Multipliers are relative to 200A, from RSMeans 091000 Electric Service (Residential Square Foot Assemblies, national average), which prices a full residential service by amperage. Only the ratios matter here -- the level is LBNL's, and the RSMeans county price index localizes costs downstream, so these stay national.

RSMeans Total:                                                  100A        $1,093 / 200A        $1,818 / 400A        $10,329  ⇒  ratios 0.60 / 1.00 / 5.68

Cross-checked against industry cost guides: 100A ~0.6,                       300A ~1.75, 400A ~2.7-5.0

- [homeguide.com](https://homeguide.com/costs/cost-to-replace-electrical-panel) (visited July 1, 2026)
- [costhelper.com](https://home.costhelper.com/electrical-upgrading-electrical-service.html) (visited July 1, 2026)
- [angi.com](https://www.angi.com/articles/ask-angie-what-does-it-cost-upgrade-200-amps.htm) (visited July 1, 2026)
- [fixr.com](https://www.fixr.com/costs/install-electrical-circuit-panel-upgrade) (visited July 1, 2026)

Per-amperage ratio choices:

- 150A ⇒ 0.80 - interpolated 100A-200A; both sources agree 100A is 0.60.
- 200A ⇒ 1.00 - anchor (LBNL median).
- 300A ⇒ 2.00 - no RSMeans 300A multiplier. A linear 200A-400A interp gives 3.34 but overshoots, since the cost curve is convex (the jump is the 400A service, not 300A). Industry guides say ~1.75, so we use 2.0.
- 400A ⇒ 4.00 - RSMeans says 5.68, guides span 2.7-5.0 (simple swap vs full service upgrade); 4.0 is a compromise.

Note that RSMeans figures are from the 2018 release.

### Weatherization

We derive upfront costs for weatherization from an [LBNL report](https://escholarship.org/content/qt0818n68p/qt0818n68p.pdf) (Table 10) by matching the each weatherization measure modeled in ResStock to a measure cost reported in the LBNL study. These cost values reflect the cost of applying the measure to an archetypal home (1,768 sq ft, 1-story, wood framed, single-family dwelling with a basement foundation, built in 1970). The matching of measures and corresponding costs is included in the table below.

| ResStock Measure | LBNL Measure | LBNL Cost | Cost per sq ft (LBNL Cost / 1,768) |
| --- | --- | --- | --- |
| Attic floor insulation to IECC levels (R-30, R-49, or R-60, depending on climate zone) | Attic floor insulation, R-60 | $4,402 | $2.49 |
| Air sealing (30% reduction in ACH-50) | Seal envelope, typical | $831 | $0.47 |
| Duct sealing (Improved to 10% leakage, R-8 insulation) | Duct seal, typical | $849 | $0.48 |
| Wall insulation (R-13 drill-and-fill insulation for homes who do not have insulation) | Drill and fill walls, R-13 | $3,135 | $1.77 |
| Foundation wall and rim joist insulation (R-10) | Foundation wall insulation, R-18 | $6,794 | $3.84 |
| Finished attic and cathedral ceiling insulation (R-30) | Roof insulation, R-35 | $13,575 | $7.68 |

The mapping of measures to weatherization packages is complex and dependent on building characteristics such as attic and foundation type. The full mapping can be found in `src/3_predict/predict_weatherization_costs.py`.

Since all LBNL costs are reported in 2019 dollars, we adjust for inflation to bring costs to the present-day. Note that the cost of wall insulation is included for all homes, regardless of whether or not they currently have insulation, so insulation costs likely represent an upper bound.

Further, since these costs are based on a median home of 1,768 sqft, we compute the per sqft costs for each measure. We then apply this per sqft cost estimate to a building’s sq ft value to produce cost estimates by measure.

We assume that R49 attic floor insulation costs the same as R60 attic floor insulation.

### Solar

We use one large data set of solar installations to estimate solar costs:

- Lawrence Berkeley National Lab [Tracking the Sun](https://emp.lbl.gov/tracking-the-sun), 3.2 million installations, 2002-2024

We start by first filtering to only focus on installations since 2017 since the cost of solar installations has fallen so dramatically over the 20 year period of the dataset that we only want to include recently installed system. This more recent sample decreases the number of installations to 2.6 million. Similar to the heat pump and heat pump water heater data, we adjust the cost of each project to represent a present-day national average by correcting for inflation and location-specific materials & labor costs. The largest predictor of a solar installation cost is the size of the installation as measured in kW. Therefore, we develop a quantile regression model that predicts solar installation cost given the expected kW size of an installation on a home. We used a quantile rather than a linear regression to capture the median project cost rather than the mean project cost, since the project cost distribution has a long upper tail.

## Location-Invariant Upfront Costs

We assume the upfront costs of stoves, dryers, and electric vehicles do not differ by location, so we do not adjust these costs by RSMeans factors. We estimate these costs using prices available online. For these upfront costs, we report an approximate lower bound on price, since the consumer has more power over how much they spend on an appliance or vehicle purchase relative to a home upgrade like a heat pump installation. The base model of an appliance or vehicle will be sufficient for most households, while some households may choose to spend more for a higher-end appliance or vehicle.

### Stoves and Dryers

Appliance upfront costs are based on desk research: costs for ER dryers, heat pump dryers, ER ranges, and induction ranges are approximate lower bounds from [Home Depot](https://www.homedepot.com/b/Appliances/N-5yc1vZbv1w), [Lowe's](https://www.lowes.com/c/Appliances), and [Best Buy](https://www.bestbuy.com/site/electronics/home-appliances/abcat0900000.c?id=abcat0900000) as of May 2026.

### EVs

We calculate the upfront costs of Electric Vehicles with the help of a tool previously created by other organizations looking to incentivize EV adoption, specifically the [EV Savings Calculator](https://ev.pge.com/vehicles) from PG&E. We apply three key filters to the dataset: (1) no used cars, (2) no cars > $125,000, and (3) no cars outside 200-400 mile range. There are six vehicle types defined in this tool: Sedans, Station Wagons, Sports Cars, Pickup Trucks, Vans, and SUVs. Each of these vehicle types is mapped to one of the four NREL ResStock EV types: Pickup, SUV, Midsize, Compact. For each of these four vehicle types, we find the 20th percentile, median, and 80th percentile vehicle by cost. These cost values are then applied to ResStock buildings based on the value of the `in_electric_vehicle_battery` column, which samples a representative EV type for the building given NREL research.

### L2 EV Chargers

Up-to-date estimates for costs of home installation of L2 chargers are difficult to find from reliable sources. We use [this study](https://theicct.org/wp-content/uploads/2021/06/ICCT_EV_Charging_Cost_20190813.pdf) from ICCT, which was written in 2019 but project costs through 2025. The study uses a 3% per year reduction in cost from 2019-2025 to estimate costs for a home L2 charger installation in 2025. Following the report methodology, we estimate the cost of an attached home installation as the average of the costs of a detached and apartment installation. We also assume a long-tailed distribution of costs, with a 20th percentile = 70% of the median and a 80th percentile = 300% of the median.

Elsewhere on the web there are many private websites listing upfront cost estimates, and the ICCT estimates fall within most of them (though they may be slightly on the low end). Given that costs vary widely depending on household wiring/panel/layout characteristics, if we want to refine these estimates further, it would be more worth it to develop different estimates based on existing panel size than to try to do extra internet research to find a different/better source or set of sources.

### Energy audit

Energy audit costs are not currently estimated in the repo. This internal [Energy Audit Costs](https://docs.google.com/spreadsheets/d/1DIX8J6nxvah7actyrgRRBscdKWgHMo1sMtnL3r_j5t4/edit#gid=1590313007) resource is available for future work.
