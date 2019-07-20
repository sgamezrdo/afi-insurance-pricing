# afi-insurance-pricing

Code used as base for a talk on "Pricing models for insurance" given at AFI. 

## Dataset
Data description from the R package. Available at: https://github.com/dutangc/CASdatasets/blob/master/pkg/man/freMTPL.Rd

freMTPLfreq (policy data) contains 10 columns:

PolicyID: The policy ID (used to link with the claims dataset).
ClaimNb: Number of claims during the exposure period.
Exposure: The period of exposure for a policy, in years.
Power: The power of the car (ordered categorical).
CarAge: The vehicle age, in years.
DriverAge: The driver age, in years (in France, people can drive a car at 18).
Brand: The car brand divided in the following groups: A- Renaut Nissan and Citroen, B- Volkswagen, Audi, Skoda and Seat, C- Opel, General Motors and Ford, D- Fiat, E- Mercedes Chrysler and BMW, F- Japanese (except Nissan) and Korean, G- other.
Gas: The car gas, Diesel or regular.
Region: The policy region in France (based on the 1970-2015 classification).
Density: The density of inhabitants (number of inhabitants per km2) in the city the driver of the car lives in.
freMTPLsev contains 2 columns:

PolicyID: The occurence date (used to link with the contract dataset).
ClaimAmount: The cost of the claim, seen as at a recent date.

## Acknowledgements
I would like to thank Luis Guerra for recommending me to give that talk, and sharing with me the contents of his previous sessions. I would also like to thank AFI, for letting me share this code. 
