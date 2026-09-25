# Phase 6 results for the question-type round (D-037): the human test set, English part

Generated on 2026-09-25 by `scripts/phase6_report.py`; do not edit by hand. Protocol: D-032, bars: D-030 in [DECISIONS.md](../DECISIONS.md).

**Test set:** 64 English messages over 10 intents, labelled by the owner (MR), SHA-256 `e6961fd244c5…`. The overlap check with all training data reports 0 overlaps. The Gulf Arabic, Arabizi and mixed parts of the test set do not exist yet, so **this evaluation covers English only**. With 64 messages, one message is 1.6 points, so the intervals are wide.

This model adds about 10,000 examples of question types the earlier models had rarely seen (status checks, how-to, conditions, costs, other countries, off-topic requests; D-037), from Bitext, MASSIVE and the local Qwen teacher. Like D-033's model, it trained on real messages from the train splits of Banking77 and CLINC150, whose test splits the English test set comes from. Earlier results: [phase6_eval.md](phase6_eval.md), [phase6b_eval.md](phase6b_eval.md); the CAMeLBERT and Qwen rows are the same systems (Qwen's zero-shot answers are reused).

## Results

| System | Intent accuracy (95% interval) | Macro-F1 | Valid JSON | Reply in the right language |
|---|---|---:|---|---|
| DanaLM (62M, question-type round) | 84.4% (54/64; 74–91%) | 85.1% | 100.0% (64/64; 94–100%) | 100.0% (64/64; 94–100%) |
| CAMeLBERT-mix classifier (110M) | 46.9% (30/64; 35–59%) | 46.4% | – | – |
| Qwen3.5-35B-A3B, zero-shot | 89.1% (57/64; 79–95%) | 90.3% | – | – |

- Messages without a usable intent, counted as wrong (D-032): DanaLM (62M, question-type round) 0; CAMeLBERT-mix classifier (110M) 0; Qwen3.5-35B-A3B, zero-shot 0. (DanaLM: an answer that is not valid JSON; Qwen: a message its answer lines did not label.)
- Macro-F1 difference, DanaLM minus CAMeLBERT-mix classifier (110M): +38.7 points (paired bootstrap 95% interval +26.7 to +51.8).
- Macro-F1 difference, DanaLM minus Qwen3.5-35B-A3B, zero-shot: -5.1 points (paired bootstrap 95% interval -16.9 to +5.7).

## The D-030 bars (English part only)

| Bar | Target | Measured | Result |
|---|---|---|---|
| Valid JSON | ≥ 99% | 100.0% (64/64; 94–100%) | **met** |
| Intent macro-F1 vs CAMeLBERT | within 3 points | +38.7 points (95% interval +26.7 to +51.8) | **met** |
| Coverage at the real-dev threshold (0.701) | ≥ 70% answered, ≥ 95% of them right | 92.2% answered (59/64), 89.8% right | **missed** |
| Reply in the right language | ≥ 95% | 100.0% (64/64; 94–100%) | **met** |

## Reply quality

Qwen judged the 64 valid DanaLM replies it gave a full verdict for on four yes/no questions: answers 93.8% (60/64; 85–98%), polite clear 90.6% (58/64; 81–96%), language 96.9% (62/64; 89–99%), safe 98.4% (63/64; 92–100%). **All four yes: 87.5% (56/64; 77–94%).** Qwen wrote DanaLM's training replies, so it may be lenient; every reply is listed at the end for the owner's spot check.

## Per intent

| Intent | Messages | DanaLM (62M, question-type round) | CAMeLBERT-mix classifier (110M) | Qwen3.5-35B-A3B, zero-shot |
|---|---:|---:|---:|---:|
| lost_or_stolen_card | 8 | 7/8 | 3/8 | 8/8 |
| account_access | 7 | 6/7 | 4/7 | 6/7 |
| fees_and_charges | 7 | 7/7 | 5/7 | 5/7 |
| transfer_issue | 7 | 6/7 | 4/7 | 6/7 |
| unrecognized_transaction | 7 | 3/7 | 0/7 | 6/7 |
| balance_or_statement | 6 | 4/6 | 0/6 | 6/6 |
| card_not_working | 6 | 6/6 | 4/6 | 5/6 |
| order_status | 6 | 6/6 | 3/6 | 6/6 |
| other | 6 | 6/6 | 6/6 | 6/6 |
| loans_and_credit | 4 | 3/4 | 1/4 | 3/4 |

## Sensitivity: without the 3 disputed messages

An independent audit disagreed with the owner's label on review rows 28, 57, 58. Leaving those messages out (the owner's labels stay the primary result): DanaLM (62M, question-type round) 86.9% (53/61; 76–93%); CAMeLBERT-mix classifier (110M) 47.5% (29/61; 36–60%); Qwen3.5-35B-A3B, zero-shot 91.8% (56/61; 82–96%).

## Every DanaLM error (10 of 64)

| Id | Message | Gold | DanaLM intent | Confidence | DanaLM reply |
|---|---|---|---|---:|---|
| t0006 | is there a problem with my account, it appears to be blocked | account_access | transfer_issue | 0.50 | I am sorry to hear about the issue with your account. Please share the specific error message you received so I can forward this to our support team. |
| t0009 | is my visa bill over my limit this month | balance_or_statement | loans_and_credit | 1.00 | I cannot view your specific account details here. Please contact our support team to discuss your available spending limit. |
| t0013 | have i gone over my entertainment budget | balance_or_statement | other | 1.00 | I cannot provide cooking instructions orlete عناءot items. Your request will be passed on to someone who can help. |
| t0020 | my dog swallowed my credit card and i need a new one | lost_or_stolen_card | card_not_working | 0.95 | I cannot process changes or process changes myself. Please contact our support team to start the process for your new card. |
| t0028 | look up the lowest amount i can pay on my electricity bill | loans_and_credit | bill_inquiry | 0.96 | Please log in to view thederbill_inquiry on our website to see the amount. |
| t0054 | You keep declining my transfers? It's always been working really well so far, but when I tried to buy something just now the card got declined. I tried couple times already but same thing. | transfer_issue | card_not_working | 0.82 | Declines can happen for various security reasons. Please try again later or call support so we can investigate this for you. |
| t0057 | im so mad right now. theres several charges that I think my x boyfriend made on my card. the companys on the website wouldn't refund me my money, they told me to contact my bank. DO something please. | unrecognized_transaction | refund_request | 0.98 | We are sorry for the confusion regarding your refund. Please share the transaction details so we can pass this to our team for review. |
| t0060 | my online bill has some items that i didn't purchase from penneys | unrecognized_transaction | bill_inquiry | 0.53 | You can find the payment options in the app or on our website. We will forward your inquiry to the team if you need more help. |
| t0061 | this charge is bs | unrecognized_transaction | fees_and_charges | 0.53 | Please share the transaction date and we will forward your query to the billing department. |
| t0062 | i think someone jacked my bank account | unrecognized_transaction | other | 0.45 | I am unable to access personal account details or personal data. I will pass your request to a team member who can help. |

## Latency and memory

Per message, batch 1, greedy decoding with the current decoder (no KV cache yet; Phase 7 adds it and quantizes). Median and p95 over the 64 messages after warm-up. CPU: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 4 threads.

| | CPU, float32 | GPU (RTX 4070), bf16 |
|---|---:|---:|
| Median per message | 0.73 s | 0.40 s |
| p95 per message | 1.07 s | 0.49 s |
| Answer tokens (mean) | 46 | 46 |
| Tokens per second | 63 | 116 |
| Memory | weights 237 MB; process peak 1199 MB (before loading the model: 922 MB) | peak allocated 371 MB |

## All replies (for the owner's spot check)

<details><summary>Show the 64 answers</summary>

| Id | Message | Gold | DanaLM | Reply | Qwen: good? |
|---|---|---|---|---|---|
| t0001 | I do not know my passcode. | account_access | account_access | If you do not recognize your passcode, please use the reset option on the login screen. Our team can guide you if you continue to face issues. | yes |
| t0002 | I need a new passcode. | account_access | account_access | You can reset your passcode by following the steps on the login page. If you need more help, please reach out to our support team. | yes |
| t0003 | Can I be given a new passcode? | account_access | account_access | You can reset your passcode through the app's login screen, or contact our support team for further assistance. | yes |
| t0004 | I lost my password | account_access | account_access | You can reset your password by following the steps on the login page of our app or website. | yes |
| t0005 | can you check what the problem with my account is, it seems to b e blocked | account_access | account_access | I cannot see your account details or the reason for the e block. Please contact our support team directly so they can investigate and assist you. | yes |
| t0006 | is there a problem with my account, it appears to be blocked | account_access | transfer_issue | I am sorry to hear about the issue with your account. Please share the specific error message you received so I can forward this to our support team. | yes |
| t0007 | did someone who is unauthorized try to get into my bank account | account_access | account_access | I am unable to view account details or specific details in this chat. Please contact our support team so they can investigate your account status. | yes |
| t0008 | what have i spent things on | balance_or_statement | balance_or_statement | I am unable to see your specific spending details here. Please check your transaction list in the app or contact our team for assistance. | yes |
| t0009 | is my visa bill over my limit this month | balance_or_statement | loans_and_credit | I cannot view your specific account details here. Please contact our support team to discuss your available spending limit. | yes |
| t0010 | tell me how much i have been buying as of late | balance_or_statement | balance_or_statement | I cannot see your specific spending details here. Please log in to your account to view your recent purchase details. | yes |
| t0011 | check my wells fargo account to see if i have enough for these nike's | balance_or_statement | balance_or_statement | You can check your available funds by logging into your account or checking your account. | yes |
| t0012 | what've i spent on gas this month | balance_or_statement | balance_or_statement | I cannot see your specific spending details here. Please check your statement or call support for the correct information. | yes |
| t0013 | have i gone over my entertainment budget | balance_or_statement | other | I cannot provide cooking instructions orlete عناءot items. Your request will be passed on to someone who can help. | no: Reply contains irrelevant text about cooking and Arabic characters, failing clarity and language requirements. |
| t0014 | Is there a trick to get the disposable virtual card to work? | card_not_working | card_not_working | We are sorry your virtual card is not working. Please contact our support team so they can check the status and assist you. | yes |
| t0015 | My contactless is not working | card_not_working | card_not_working | Please reach out to our support team so they can check your account status and help you resolve this. | yes |
| t0016 | I was getting cash and can't get my card back. | card_not_working | card_not_working | If your card was retained by an ATM, please contact our support team immediately for assistance. They will guide you on the next steps to retrieve it. | yes |
| t0017 | Is there a way to know when my card will arrive? | card_not_working | card_not_working | Delivery times vary depending on your location and card type. Please contact our support team to check the status of your delivery. | yes |
| t0018 | i was trying to use my card to get food at publix but it got declined | card_not_working | card_not_working | I am sorry to hear your card was declined. Please contact our support team so they can investigate this for you. | yes |
| t0019 | why my card stop working | card_not_working | card_not_working | This requires a manual review by our specialists. Please contact us so we can pass your case to the team. | yes |
| t0020 | my dog swallowed my credit card and i need a new one | lost_or_stolen_card | card_not_working | I cannot process changes or process changes myself. Please contact our support team to start the process for your new card. | no: Reply contains a redundant phrase 'process changes or process changes' making it poorly written. |
| t0021 | A fee was charged for my recent withdrawal. | fees_and_charges | fees_and_charges | Withdrawal fees are applied based on account type and account conditions. We will forward your query to the relevant department for a detailed explanation. | yes |
| t0022 | I got a fee for an ATM withdrawal. | fees_and_charges | fees_and_charges | ATM withdrawal fees depend on the ATM location and your account type. Please contact us with the transaction date so we can pass this to the relevant team. | yes |
| t0023 | I thought transfers were free, why was I charged a fee? | fees_and_charges | fees_and_charges | Fees depend on the transfer method and account type you hold. Please share the transaction details so we can investigate this with our team. | yes |
| t0024 | Are there extra charges for exchanging currency? | fees_and_charges | fees_and_charges | Currency exchange rates may involve additional charges depending on the transaction. Our team will provide the current details of these charges once you share the details. | yes |
| t0025 | when i go to cuba will there be a transaction fee for using my discover card | fees_and_charges | fees_and_charges | I do not have access to view your specific transaction details. Please contact our support team to get the exact fee applicable to your Discover card. | yes |
| t0026 | what's my checking account yielding in interest | fees_and_charges | fees_and_charges | Interest rates depend on your account tier and the interest rate. Please contact our support team to get the accurate rate for your account. | yes |
| t0027 | what does my checking account at chase earn in interest | fees_and_charges | fees_and_charges | I do not have access to your account information to check Chase directly. Please contact our support team for assistance with your Chase website. | yes |
| t0028 | look up the lowest amount i can pay on my electricity bill | loans_and_credit | bill_inquiry | Please log in to view thederbill_inquiry on our website to see the amount. | no: Contains a typo ('thedderbill_inquiry') and is unclear. |
| t0029 | could i perhaps increase by credit limit by 500 dollars | loans_and_credit | loans_and_credit | Please submit a request via the app or our website to review your credit limit. | yes |
| t0030 | what's my spending limit on my discover | loans_and_credit | loans_and_credit | I am unable to see your current spending limit details. Please contact our support team or visit a branch for assistance with this information. | yes |
| t0031 | has my application for the hsbc card been approved yet | loans_and_credit | loans_and_credit | I cannot check the status of your application directly. Our team will review your query and get back to you shortly. | yes |
| t0032 | My bags were stolen. I need a new card but need to cancel the stolen one. | lost_or_stolen_card | lost_or_stolen_card | Please reach out to our customer service team to discuss canceling your stolen card. They will provide the necessary steps to ensure your account is safe andاصابي. | no: Contains non-English characters 'اصابي' |
| t0033 | I cant find my card, it's gone. | lost_or_stolen_card | lost_or_stolen_card | Please reach out to our support team right away so they can assist you with blocking the card. | yes |
| t0034 | I cannot find my credit card. | lost_or_stolen_card | lost_or_stolen_card | Please contact our support team immediately to report your missing card. They will guide you through the necessary steps to secure your account. | yes |
| t0035 | How do I report my card stolen? | lost_or_stolen_card | lost_or_stolen_card | To report a stolen card, please contact our dedicated team right away. They will assist you with blocking the card and arranging a new one. | yes |
| t0036 | please tell me how to go about reporting a lost card | lost_or_stolen_card | lost_or_stolen_card | Please contact our support team directly to report your lost card and get assistance. | yes |
| t0037 | set up a report for a lost card | lost_or_stolen_card | lost_or_stolen_card | Please report the lost card through the app or on our website to block it. Our team will review your request and contact you shortly. | yes |
| t0038 | i need to report that someone ran off with my td bank card earlier | lost_or_stolen_card | lost_or_stolen_card | Please report this incident through the app or website to block your card. Our team will review your case and contact you shortly. | yes |
| t0039 | can you monitor my package's progress | order_status | order_status | I cannot see your specific order details directly. Please share this with the relevant team for an update. | yes |
| t0040 | can you help with delivery tracking | order_status | order_status | Please check the app or website for details on tracking tracking. We will pass your request to the relevant team. | no: Contains a typo 'tracking tracking' |
| t0041 | track my order | order_status | order_status | Please provide your order number so I can pass this to the team for assistance. | yes |
| t0042 | what is the status of the order i placed with you yesterday | order_status | order_status | I do not have access to view the current status of your order. Please check your email or contact the restaurant for the latest update. | yes |
| t0043 | has my package been sent | order_status | order_status | I cannot see the current status of your package directly. Please contact our support team so they can investigate this for you. | yes |
| t0044 | i need a delivery tracked | order_status | order_status | You can track your delivery status directly in the app or on our website. We will forward your request to the team for further assistance. | yes |
| t0045 | What are the most current exchange rates? | other | other | I will pass your question about exchange rates to a specialist who can assist you further. | yes |
| t0046 | what are exchange rates | other | other | I will forward your inquiry about exchange rates to a specialist who can provide the details. | yes |
| t0047 | ai is a helpful item to have in my home, thank you | other | other | Thank you for your kind words. Please let me know if you need any further assistance. | yes |
| t0048 | enjoy your day, goodbye | other | other | Goodbye! Have a wonderful day and take care. | yes |
| t0049 | My transfer did not go through. | transfer_issue | transfer_issue | I am sorry to hear that your transfer did not go through. Please contact our support team so they can investigate the status for you. | yes |
| t0050 | Why hasn't my transfer gone through yet? | transfer_issue | transfer_issue | We are sorry to hear about the delay with your transfer. Please share the reference number so we can pass this to our team for review. | yes |
| t0051 | I am trying to make a transfer and am unsuccessful, can you please tell me why? | transfer_issue | transfer_issue | I am sorry to hear about the issue with your transfer. Please share the specific error message or reference so I can pass this to our team for review. | yes |
| t0052 | Please explain why the transfers shows up as pending? | transfer_issue | transfer_issue | Please share the details of the transfer so we can pass this to our team. | yes |
| t0053 | Why didn't my balance change after I transferred some money? | transfer_issue | transfer_issue | Transfers may take time to reflect in the recipient's account depending on the banks involved. Please provide the transaction reference so we can check further. | no: Invents a general policy explanation about transfer times without verifying the specific account status. |
| t0054 | You keep declining my transfers? It's always been working really well so far, but when I tried to buy something just now the card got declined. I tried couple times already but same thing. | transfer_issue | card_not_working | Declines can happen for various security reasons. Please try again later or call support so we can investigate this for you. | yes |
| t0055 | transaction failed? | transfer_issue | transfer_issue | We are forwarding your concern to the relevant team for review. Please wait for them to contact you with next steps. | yes |
| t0056 | Why was I charged multiple times for the same thing? | unrecognized_transaction | unrecognized_transaction | Please report this duplicate charge to our support team so they can investigate the duplicate charge for you. | yes |
| t0057 | im so mad right now. theres several charges that I think my x boyfriend made on my card. the companys on the website wouldn't refund me my money, they told me to contact my bank. DO something please. | unrecognized_transaction | refund_request | We are sorry for the confusion regarding your refund. Please share the transaction details so we can pass this to our team for review. | yes |
| t0058 | Hello. I checked my transactions and saw I was double charged on my card for a restaurant visit. I would like to get my money back. | unrecognized_transaction | unrecognized_transaction | We recommend contacting our support team immediately to review your account details. They will guide you on the next steps to resolve this issue. | yes |
| t0059 | There is a payment that is not mine in the app. Please advise/ | unrecognized_transaction | unrecognized_transaction | Please report this payment immediately through the app or contact our fraud team so they can investigate the transaction. They will guide you on the next steps to secure your account. | yes |
| t0060 | my online bill has some items that i didn't purchase from penneys | unrecognized_transaction | bill_inquiry | You can find the payment options in the app or on our website. We will forward your inquiry to the team if you need more help. | no: Does not address the customer's specific concern about unauthorized items; provides irrelevant information about payment options. |
| t0061 | this charge is bs | unrecognized_transaction | fees_and_charges | Please share the transaction date and we will forward your query to the billing department. | yes |
| t0062 | i think someone jacked my bank account | unrecognized_transaction | other | I am unable to access personal account details or personal data. I will pass your request to a team member who can help. | yes |
| t0063 | ok good chatting goodbye | other | other | Goodbye! Have a wonderful day and take care. | yes |
| t0064 | what is albert einstein famous for | other | other | I do not have access to current algements. I will pass this question to a team member who can assist you further. | no: Reply contains typo 'algements' and incorrectly claims inability to answer general knowledge questions rather than clarifying chatbot limitations. |

</details>
