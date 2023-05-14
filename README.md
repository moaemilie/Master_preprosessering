# Preprosessering av hyperspektrale bilder

Filene inneholder preprosesseringen brukt i masteren. Funkjsonene i begge filene gjør følgende:
- Hvitreferanse
- ROI
- Lager maske
- Savitzky-Golay filter
- Gjør om til absorbans
- Henter ut 12 gjennomsnittsspektre fra hvert bilde, med maske.

Filene skiller mellom hvilken scatter correction som er brukt. 

- I preprosessering_SNV ligger koden brukt til preprosessering med SNV
- I preprosessering_MSC ligger koden brukt til preprosessering med MCS
