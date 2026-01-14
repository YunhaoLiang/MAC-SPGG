"""
Prompt templates for summary evaluation.
"""

# Few-shot example for evaluation prompt
FEW_SHOT_EXAMPLE = """Example 1:
Document: (CNN)Donald Sterling's racist remarks cost him an NBA team last year. But now it's his former female companion who has lost big. A Los Angeles judge has ordered V. Stiviano to pay back more than $2.6 million in gifts after Sterling's wife sued her. In the lawsuit, Rochelle "Shelly" Sterling accused Stiviano of targeting extremely wealthy older men. She claimed Donald Sterling used the couple's money to buy Stiviano a Ferrari, two Bentleys and a Range Rover, and that he helped her get a $1.8 million duplex. Who is V. Stiviano? Stiviano countered that there was nothing wrong with Donald Sterling giving her gifts and that she never took advantage of the former Los Angeles Clippers owner, who made much of his fortune in real estate. Shelly Sterling was thrilled with the court decision Tuesday, her lawyer told CNN affiliate KABC. "This is a victory for the Sterling family in recovering the $2,630,000 that Donald lavished on a conniving mistress," attorney Pierce O'Donnell said in a statement. "It also sets a precedent that the injured spouse can recover damages from the recipient of these ill-begotten gifts." Stiviano's gifts from Donald Sterling didn't just include uber-expensive items like luxury cars. According to the Los Angeles Times, the list also includes a $391 Easter bunny costume, a $299 two-speed blender and a $12 lace thong. Donald Sterling's downfall came after an audio recording surfaced of the octogenarian arguing with Stiviano. In the tape, Sterling chastises Stiviano for posting pictures on social media of her posing with African-Americans, including basketball legend Magic Johnson. "In your lousy f**ing Instagrams, you don't have to have yourself with -- walking with black people," Sterling said in the audio first posted by TMZ. He also tells Stiviano not to bring Johnson to Clippers games and not to post photos with the Hall of Famer so Sterling's friends can see. "Admire him, bring him here, feed him, f**k him, but don't put (Magic) on an Instagram for the world to have to see so they have to call me," Sterling said. NBA Commissioner Adam Silver banned Sterling from the league, fined him $2.5 million and pushed through a charge to terminate all of his ownership rights in the franchise. Fact check: Donald Sterling's claims vs. reality CNN's Dottie Evans contributed to this report.

Summary: donald sterling , nba team last year . sterling 's wife sued for $ 2.6 million in gifts . sterling says he is the former female companion who has lost the . sterling has ordered v. stiviano to pay back $ 2.6 m in gifts after his wife sued . sterling also includes a $ 391 easter bunny costume , $ 299 and a $ 299 .

Evaluation scores:
Relevance: 1.6667
Coherence: 1.3333
Consistency: 1
Fluency: 1"""


EVALUATION_PROMPT_TEMPLATE = """Evaluate the following summary based on the given document. Provide scores in the range [0, 5] for each dimension.

{few_shot}


Now evaluate the following:
Document: {document}

Summary: {summary}

Now, output scores ONLY:
Relevance: {relevance}
Coherence: {coherence}
Consistency: {consistency}
Fluency: {fluency}"""


def create_structured_prompt(
    text: str,
    summary: str,
    r_relevance: float = None,
    r_coherence: float = None,
    r_consistency: float = None,
    r_fluency: float = None
) -> str:
    """
    Create structured evaluation prompt with optional score labels.
    
    For training: Include all four scores in the output template.
    For inference: Leave score fields empty for model to generate.
    
    Args:
        text: Source document text.
        summary: Summary to evaluate.
        r_relevance: Relevance score (None for inference).
        r_coherence: Coherence score (None for inference).
        r_consistency: Consistency score (None for inference).
        r_fluency: Fluency score (None for inference).
    
    Returns:
        Formatted prompt string.
    """
    if r_relevance is not None:
        # Training format with ground truth scores
        prompt = EVALUATION_PROMPT_TEMPLATE.format(
            few_shot=FEW_SHOT_EXAMPLE,
            document=text,
            summary=summary,
            relevance=f"{r_relevance:.1f}",
            coherence=f"{r_coherence:.1f}",
            consistency=f"{r_consistency:.1f}",
            fluency=f"{r_fluency:.1f}"
        )
    else:
        # Inference format without scores
        prompt = EVALUATION_PROMPT_TEMPLATE.format(
            few_shot=FEW_SHOT_EXAMPLE,
            document=text,
            summary=summary,
            relevance="",
            coherence="",
            consistency="",
            fluency=""
        )
    
    return prompt
