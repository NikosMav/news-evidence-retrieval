# Qualitative failure cases (auto-sampled from eval detail)

These examples come from the ISOT title-recovery sanity check (query = article title, gold = that article's passages).
They are misses against finding the source article, not against finding evidence for a claim.

## Misses (gold article absent from top-5)

- **Query title:** 'OOPS! New App Allows Users To Remain ANONYMOUS…Defies Liberal Media Narrative…Shows TRUMP Winning BIG Over Crooked Hillary'  
  Gold bucket: `fake` / subject `left-news`.  
  Top-1 returned: "'Is a Tweet policy?' State Department officials ponder" (bucket `true`, score=0.029).
- **Query title:** 'WOMAN CRIES After Seeing How Easily Our Votes Are Stolen By Electronic Voting Machines [VIDEO]'  
  Gold bucket: `fake` / subject `Government News`.  
  Top-1 returned: 'STEALING THE ELECTION…SHOCKING CBS Report Shows How Anyone Can Hack The Vote For $15…VIDEO Shows How Easy It Is To Rig The System' (bucket `fake`, score=0.032).
- **Query title:** 'HOLY MOLY! TRUMP GIVES EPIC News Conference…SLAYS PRESS…Dresses Down CNN…”Your Ratings Are Lower Than Congress…Network Is All About “Hate” [VIDEO]'  
  Gold bucket: `fake` / subject `left-news`.  
  Top-1 returned: 'Trump Announces That Surrogates Won’t Appear On Networks That Don’t ‘Promote’ His Agenda' (bucket `fake`, score=0.029).
- **Query title:** 'DONALD TRUMP DITCHES THE PRESS To Do Something Really Fun…And Normal'  
  Gold bucket: `fake` / subject `politics`.  
  Top-1 returned: 'Trump Just Had The Most AWKWARD Moment With Saudi Crown Prince, HUMILIATES Himself (VIDEO)' (bucket `fake`, score=0.029).
- **Query title:** 'What This Trump Company Did To A Veteran Will Disgust You'  
  Gold bucket: `fake` / subject `News`.  
  Top-1 returned: 'WOW! US Marine And Navy Veteran Writes BLISTERING Open Letter To Khizr Khan: “Does it matter whether Mr. Trump has ‘sacrificed’? Has Ms. Clinton ‘sacrificed’ for this nation? How about Mr. Obama?”' (bucket `fake`, score=0.026).
- **Query title:** 'OHIO ELECTOR TORCHES Anti-Trump Letters He Received From Crybaby Liberals [Video]'  
  Gold bucket: `fake` / subject `left-news`.  
  Top-1 returned: 'MISLEADING MAINSTREAM MEDIA Is Pushing False Narrative That Trump Electors Could Steal Election From Him …Why It’s Not Gonna Happen…And Why Their Attack On Our Democracy Is A Really BAD Idea' (bucket `fake`, score=0.032).
- **Query title:** 'HORRIBLE! TOP DEMOCRATS Refuse To Stand For Gold Star Widow [Video]'  
  Gold bucket: `fake` / subject `politics`.  
  Top-1 returned: 'HARSH AND TRUE! TOP TEN Reasons Obama Was The WORST President EVER! [Video]' (bucket `fake`, score=0.023).
- **Query title:** 'FACTBOX - German coalition watch: Agreeing on lowest common denominator not enough - Greens'  
  Gold bucket: `true` / subject `worldnews`.  
  Top-1 returned: "Germany's Greens all but rule out three-way 'Jamaica' coalition" (bucket `true`, score=0.032).

## Opposite-label neighbors in top-5 (source-bucket leakage)

- **Query:** 'OOPS! New App Allows Users To Remain ANONYMOUS…Defies Liberal Media Narrative…Shows TRUMP Winning BIG Over Crooked Hillary' (`fake`) → 3 opposite-label hits in top-5; top-1 `true`: "'Is a Tweet policy?' State Department officials ponder".
- **Query:** 'WOMAN CRIES After Seeing How Easily Our Votes Are Stolen By Electronic Voting Machines [VIDEO]' (`fake`) → 2 opposite-label hits in top-5; top-1 `fake`: 'STEALING THE ELECTION…SHOCKING CBS Report Shows How Anyone Can Hack The Vote For $15…VIDEO Shows How Easy It Is To Rig The System'.
- **Query:** 'FORMER DNC AIDES In Hot Water Over Possible Money Laundering Scheme to Fund Terrorism Overseas' (`fake`) → 2 opposite-label hits in top-5; top-1 `fake`: 'FORMER DNC AIDES In Hot Water Over Possible Money Laundering Scheme to Fund Terrorism Overseas'.
- **Query:** 'Germany says worried about new generation of Islamic State recruits' (`true`) → 2 opposite-label hits in top-5; top-1 `true`: 'Germany says worried about new generation of Islamic State recruits'.
- **Query:** 'Trump says U.S. should mull more racial profiling after Orlando shooting' (`true`) → 2 opposite-label hits in top-5; top-1 `true`: 'Trump says U.S. should mull more racial profiling after Orlando shooting'.
- **Query:** 'A Monkey Knocked Out Kenya’s Power Grid, So Conservatives Called Obama a N***** (SCREENSHOTS)' (`fake`) → 2 opposite-label hits in top-5; top-1 `fake`: 'A Monkey Knocked Out Kenya’s Power Grid, So Conservatives Called Obama a N***** (SCREENSHOTS)'.
- **Query:** 'Sister of NY attack suspect says he may have been brainwashed; appeals to Trump' (`true`) → 3 opposite-label hits in top-5; top-1 `true`: 'Sister of NY attack suspect says he may have been brainwashed; appeals to Trump'.
- **Query:** 'Rice chides Trump for criticism of judges, media' (`true`) → 3 opposite-label hits in top-5; top-1 `true`: 'Rice chides Trump for criticism of judges, media'.

Retrieved opposite-bucket neighbors show topical/style overlap across source classes. They do **not** prove or refute the query claim.

## Same-subject / outlet-style collapse

- **Query:** 'Puerto Rico needs restructuring to avoid cascading defaults: Treasury' (subject `politicsNews`) → 3 other same-subject articles in top-5.
- **Query:** 'Hillary Responds To Bernie Calling Her Unqualified, And It’s Kinda Perfect' (subject `News`) → 2 other same-subject articles in top-5.
- **Query:** 'China lodges protest after Trump call with Taiwan president' (subject `politicsNews`) → 2 other same-subject articles in top-5.
- **Query:** 'Internet Collectively Cringes After Photographer Snaps Pic Of Trump’s Nutsack' (subject `News`) → 2 other same-subject articles in top-5.
- **Query:** 'Trump Supporter Disgustingly Tells Black People To ‘Go Back To Africa’ (VIDEO)' (subject `News`) → 3 other same-subject articles in top-5.
- **Query:** 'UK PM may says looks forward to working with Trump, building ties' (subject `politicsNews`) → 2 other same-subject articles in top-5.
- **Query:** "Ireland says 'lot of work' needed to move to next phase of Brexit talks" (subject `worldnews`) → 4 other same-subject articles in top-5.
- **Query:** 'RAND PAUL: SOMEBODY WAS SPYING On Trump Campaign…It’s “already been proven to be true” [VIDEO]' (subject `politics`) → 2 other same-subject articles in top-5.

Same-subject neighbors often share wire diction or political framing; the ranker can collapse to outlet/topic style rather than the specific claim.

## Takeaway

High scores here mean the index can find an article's own passages from its title. That is a pipeline sanity check, not evidence retrieval. ISOT labels remain source buckets.
