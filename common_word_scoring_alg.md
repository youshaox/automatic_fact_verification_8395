1. preprocess the wiki titles (replace '_',  其实不用去括号里的东西)

2. Contruct the inverted index based on preprocessed wiki titles.

   'The' ------> [doc_id$_{n1}​$, …, doc\_id_${nk}​$]

   'Univeristy' ———>  [doc_id$_{m1}$, …, doc\_id_${mk}$]

3. 

   1. Claim converted to token set.

   For example, claim "I went to the Melbourne University in Australia." will be {I, went, to, the, University, of, Melbourne, in, Australia, .}.

   2. I -> posting list1, went -> posting list2, …., Australia -> posting list9. (May disregard the stop words!)

4. Scoring based on all the wiki titles in the posting list with respect to the claim "I went to the University of Melbourne in Australia." (**The usage of title based inverted index avods us to search all the wiki titles**)

   **Target:** The target wiki titles we would like to find is the "Unveristy of Melbourne" and "Australia"

   **Formula:** score =  $p \times \frac{num_{com}}{lengthOfWikiTitles}​$

   * $num_{com}$ are the common tokens between wiki titles and claim.
   * ${lengthOfWikiTitles}$ is the number of tokens in wiki title.
   * $p$ is the hyperparameter. 
     * currently we using p=1
     * Possible Improvement:
       1. We could give higher weight to longer wiki titles. (Grid search)
       2. some weights which is some combination of length of wiki titles, claim lengths or number of stopwords
       3. contextual linear predictor 

   1. For wiki title: "Melbourne Univserity" we get the score = 2/2=1
   2. For wiki title: "The", "in", we get the full score = 1/1 = 1
   3. For wiki title: "List of university hospitals " score = 1/4
   4. For wiki title: "Melbourne", "Australia" score = 1/1 =1 
   5. For wiki title: "University of Sydney" score =  2/3 =  0.67 (We could penelize it by lower p weight.)

   **Filtering**

   Asumming we get all the matching titles: "University of Melbourne", "The", "List of university hospitals ", "Melbourne" , "University of Sydney", "in", "Australia".

   1. Remove the title contained by other titles: "Melbourne" is contained by "University of Melbourne". So we removed the "Melbourne".
   2. Remove the stop words in the wiki titles: "The", "in", is removed.
   3. Ranking the remaining wiki titles.
      1. we will get:
         1. University of Melbourne (1) (WE WANTED)
         2. Australia (1) ((WE WANTED))
         3. University of Sydney (2/3) (Future we could penalise it.)
         4. List of university hospitals (2/4)

5. Compared with our Prefix Tree based searching (TBST), the common word scoring method could not only find all the results found by TBST, but also find more related wiki titles. There are patterns that TBST fails to capture compared with common word scoring when common word scoring could find all the wiki titles which could be found by TBST.

   1. Take name as an example.  (First name, Middle name, Given name)
      1. claim: "John Kennedy is the 35th president of the United States."
      2. wiki title: "John Fitzgerald Kennedy"

   2. In this situation, the TBST will fail to find the wiki title "John Fitzgerald Kennedy". However, common word scoring gives the score 2/3. We could also improve this score by weighting.

**Even though during our competition, this two methods all give the similar recall rate 0.89 with average 19 documet titles. But theoretically, the common word scoring method should have much higher recall and precision.**
