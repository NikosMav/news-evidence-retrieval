# SciFact misses (beir/scifact/test)

Drawn from the frozen BM25 / MiniLM / RRF run. A miss is nDCG@10 of 0: no relevant abstract in the top 10.

### BM25 misses a claim that MiniLM ranks

Claim `238`: Cells undergoing methionine restriction may activate miRNAs.

Gold doc ids: 2251426

- bm25 nDCG@10=0.000; top doc `4463811` Low methionine ingestion by rats extends life span.: Low methionine ingestion by rats extends life span. Dietary energy restriction has been a widely used means of experimentally extending mammalian life span. We report here that lifelong reduction in the concentration of a single dietary component, the essential amino acid L-methionine, from 0.86 to 0.17% of the diet re
- dense nDCG@10=0.631; top doc `2619579` The widespread regulation of microRNA biogenesis, function and decay: The widespread regulation of microRNA biogenesis, function and decay MicroRNAs (miRNAs) are a large family of post-transcriptional regulators of gene expression that are ∼21 nucleotides in length and control many developmental and cellular processes in eukaryotic organisms. Research during the past decade has identifie
- hybrid nDCG@10=0.431; top doc `4463811` Low methionine ingestion by rats extends life span.: Low methionine ingestion by rats extends life span. Dietary energy restriction has been a widely used means of experimentally extending mammalian life span. We report here that lifelong reduction in the concentration of a single dietary component, the essential amino acid L-methionine, from 0.86 to 0.17% of the diet re

### MiniLM misses a claim that BM25 ranks first

Claim `48`: A total of 1,000 people in the UK are asymptomatic carriers of vCJD infection.

Gold doc ids: 13734012

- bm25 nDCG@10=1.000; top doc `13734012` Prevalent abnormal prion protein in human appendixes after bovine spongiform encephalopathy epizootic: large scale survey: Prevalent abnormal prion protein in human appendixes after bovine spongiform encephalopathy epizootic: large scale survey OBJECTIVES To carry out a further survey of archived appendix samples to understand better the differences between existing estimates of the prevalence of subclinical infection with prions after the
- dense nDCG@10=0.000; top doc `18617259` Research Letters: Research Letters We report a case of preclinical variant Creutzfeldt-Jakob disease (vCJD) in a patient who died from a non-neurological disorder 5 years after receiving a blood transfusion from a donor who subsequently developed vCJD. Protease-resistant prion protein (PrP(res)) was detected by western blot, paraffin-em
- hybrid nDCG@10=0.356; top doc `18617259` Research Letters: Research Letters We report a case of preclinical variant Creutzfeldt-Jakob disease (vCJD) in a patient who died from a non-neurological disorder 5 years after receiving a blood transfusion from a donor who subsequently developed vCJD. Protease-resistant prion protein (PrP(res)) was detected by western blot, paraffin-em

### BM25, MiniLM, and hybrid all miss the top 10

Claim `13`: 5% of perinatal mortality is due to low birth weight.

Gold doc ids: 1606628

- bm25 nDCG@10=0.000; top doc `1263446` Determinants of neonatal mortality in Indonesia: Determinants of neonatal mortality in Indonesia BACKGROUND Neonatal mortality accounts for almost 40 per cent of under-five child mortality, globally. An understanding of the factors related to neonatal mortality is important to guide the development of focused and evidence-based health interventions to prevent neonata
- dense nDCG@10=0.000; top doc `7662395` Perinatal mortality in rural China: retrospective cohort study.: Perinatal mortality in rural China: retrospective cohort study. OBJECTIVES To explore the use of local civil registration data to assess the perinatal mortality in a typical rural county in a less developed province in China, 1999-2000. DESIGN Retrospective cohort study. Pregnancies in a cohort of women followed from r
- hybrid nDCG@10=0.000; top doc `7662395` Perinatal mortality in rural China: retrospective cohort study.: Perinatal mortality in rural China: retrospective cohort study. OBJECTIVES To explore the use of local civil registration data to assess the perinatal mortality in a typical rural county in a less developed province in China, 1999-2000. DESIGN Retrospective cohort study. Pregnancies in a cohort of women followed from r
