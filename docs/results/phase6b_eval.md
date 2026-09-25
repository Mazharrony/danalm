# Phase 6 results for the real-message round (D-033): the human test set, English part

Generated on 2026-09-25 by `scripts/phase6_report.py`; do not edit by hand. Protocol: D-032, bars: D-030 in [DECISIONS.md](../DECISIONS.md).

**Test set:** 64 English messages over 10 intents, labelled by the owner (MR), SHA-256 `e6961fd244c5…`. The overlap check with all training data reports 0 overlaps. The Gulf Arabic, Arabizi and mixed parts of the test set do not exist yet, so **this evaluation covers English only**. With 64 messages, one message is 1.6 points, so the intervals are wide.

This model was also trained on real messages from the train splits of Banking77 and CLINC150 (D-033), the same datasets whose test splits the English test set comes from. Near copies of the test messages were removed from training, but part of the English gain comes from the shared style. The Gulf Arabic, Arabizi and mixed test parts remain the real check. Phase 6's first result is in [phase6_eval.md](phase6_eval.md); the CAMeLBERT and Qwen rows are the same systems (Qwen's zero-shot answers are reused from that run, the same requests).

## Results

| System | Intent accuracy (95% interval) | Macro-F1 | Valid JSON | Reply in the right language |
|---|---|---:|---|---|
| DanaLM (62M, real-message round) | 84.4% (54/64; 74–91%) | 85.5% | 100.0% (64/64; 94–100%) | 100.0% (64/64; 94–100%) |
| CAMeLBERT-mix classifier (110M) | 46.9% (30/64; 35–59%) | 46.4% | – | – |
| Qwen3.5-35B-A3B, zero-shot | 89.1% (57/64; 79–95%) | 90.3% | – | – |

- Messages without a usable intent, counted as wrong (D-032): DanaLM (62M, real-message round) 0; CAMeLBERT-mix classifier (110M) 0; Qwen3.5-35B-A3B, zero-shot 0. (DanaLM: an answer that is not valid JSON; Qwen: a message its answer lines did not label.)
- Macro-F1 difference, DanaLM minus CAMeLBERT-mix classifier (110M): +39.1 points (paired bootstrap 95% interval +26.6 to +53.0).
- Macro-F1 difference, DanaLM minus Qwen3.5-35B-A3B, zero-shot: -4.7 points (paired bootstrap 95% interval -16.0 to +6.0).

## The D-030 bars (English part only)

| Bar | Target | Measured | Result |
|---|---|---|---|
| Valid JSON | ≥ 99% | 100.0% (64/64; 94–100%) | **met** |
| Intent macro-F1 vs CAMeLBERT | within 3 points | +39.1 points (95% interval +26.6 to +53.0) | **met** |
| Coverage at the real-dev threshold (0.592) | ≥ 70% answered, ≥ 95% of them right | 95.3% answered (61/64), 86.9% right | **missed** |
| Reply in the right language | ≥ 95% | 100.0% (64/64; 94–100%) | **met** |

## Reply quality

Qwen judged the 64 valid DanaLM replies it gave a full verdict for on four yes/no questions: answers 95.3% (61/64; 87–98%), polite clear 96.9% (62/64; 89–99%), language 100.0% (64/64; 94–100%), safe 98.4% (63/64; 92–100%). **All four yes: 92.2% (59/64; 83–97%).** Qwen wrote DanaLM's training replies, so it may be lenient; every reply is listed at the end for the owner's spot check.

## Per intent

| Intent | Messages | DanaLM (62M, real-message round) | CAMeLBERT-mix classifier (110M) | Qwen3.5-35B-A3B, zero-shot |
|---|---:|---:|---:|---:|
| lost_or_stolen_card | 8 | 6/8 | 3/8 | 8/8 |
| account_access | 7 | 7/7 | 4/7 | 6/7 |
| fees_and_charges | 7 | 6/7 | 5/7 | 5/7 |
| transfer_issue | 7 | 7/7 | 4/7 | 6/7 |
| unrecognized_transaction | 7 | 4/7 | 0/7 | 6/7 |
| balance_or_statement | 6 | 4/6 | 0/6 | 6/6 |
| card_not_working | 6 | 6/6 | 4/6 | 5/6 |
| order_status | 6 | 6/6 | 3/6 | 6/6 |
| other | 6 | 5/6 | 6/6 | 6/6 |
| loans_and_credit | 4 | 3/4 | 1/4 | 3/4 |

## Sensitivity: without the 3 disputed messages

An independent audit disagreed with the owner's label on review rows 28, 57, 58. Leaving those messages out (the owner's labels stay the primary result): DanaLM (62M, real-message round) 85.2% (52/61; 74–92%); CAMeLBERT-mix classifier (110M) 47.5% (29/61; 36–60%); Qwen3.5-35B-A3B, zero-shot 91.8% (56/61; 82–96%).

## Every DanaLM error (10 of 64)

| Id | Message | Gold | DanaLM intent | Confidence | DanaLM reply |
|---|---|---|---|---:|---|
| t0009 | is my visa bill over my limit this month | balance_or_statement | loans_and_credit | 0.91 | I cannot view your specific account details here. Please log in to your online banking or mobile app to view your current credit limit. |
| t0013 | have i gone over my entertainment budget | balance_or_statement | other | 0.72 | I cannot provide feedback onwellowing your entertainment budget. Your request will be passed to a team member who can help. |
| t0020 | my dog swallowed my credit card and i need a new one | lost_or_stolen_card | card_not_working | 0.98 | Please contact our support team directly so they can check your card status and assist you. |
| t0027 | what does my checking account at chase earn in interest | fees_and_charges | balance_or_statement | 0.52 | I do not have access to view personal account balances or Chase's account balances. Please log in to your online banking or mobile app to check your current balance. |
| t0028 | look up the lowest amount i can pay on my electricity bill | loans_and_credit | bill_inquiry | 0.98 | Please log in to your account to view the latest amount. Our team will review your account details if you contact them. |
| t0038 | i need to report that someone ran off with my td bank card earlier | lost_or_stolen_card | card_not_working | 0.63 | I cannot process requested card issues directly. Please contact our support team to report this issue for further assistance. |
| t0047 | ai is a helpful item to have in my home, thank you | other | missing_or_wrong_item | 0.76 | We apologize for the mix-up. Please share your order number so we can pass this to our team. |
| t0057 | im so mad right now. theres several charges that I think my x boyfriend made on my card. the companys on the website wouldn't refund me my money, they told me to contact my bank. DO something please. | unrecognized_transaction | refund_request | 0.70 | We are sorry for the confusion. Please share your transaction details so we can pass your request to the team. |
| t0060 | my online bill has some items that i didn't purchase from penneys | unrecognized_transaction | card_not_working | 0.43 | I am sorry to hear about the issue with your online bill. Please let me know which items you need so I can pass this to the team. |
| t0062 | i think someone jacked my bank account | unrecognized_transaction | account_access | 0.99 | I am sorry to hear about the trouble with your account. Please contact our support team so they can assist you with the next steps. |

## Latency and memory

Per message, batch 1, greedy decoding with the current decoder (no KV cache yet; Phase 7 adds it and quantizes). Median and p95 over the 64 messages after warm-up. CPU: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 4 threads.

| | CPU, float32 | GPU (RTX 4070), bf16 |
|---|---:|---:|
| Median per message | 1.95 s | 1.07 s |
| p95 per message | 2.81 s | 1.33 s |
| Answer tokens (mean) | 47 | 47 |
| Tokens per second | 23 | 44 |
| Memory | weights 237 MB; process peak 1195 MB (before loading the model: 927 MB) | peak allocated 371 MB |

## All replies (for the owner's spot check)

<details><summary>Show the 64 answers</summary>

| Id | Message | Gold | DanaLM | Reply | Qwen: good? |
|---|---|---|---|---|---|
| t0001 | I do not know my passcode. | account_access | account_access | You can reset your passcode by selecting the 'Forgot Passcode' option on the login screen. If you need help, we will pass your request to a specialist. | yes |
| t0002 | I need a new passcode. | account_access | account_access | You can reset your passcode by following the instructions on the login screen. If you need further help, we will pass your request to a specialist. | yes |
| t0003 | Can I be given a new passcode? | account_access | account_access | You can reset your passcode by selecting the appropriate option on the login screen. If you need further help, we will pass your request to a specialist. | yes |
| t0004 | I lost my password | account_access | account_access | You can reset your lost password by selecting the 'Forgot Password' link on the login screen. If you continue to have issues, please contact our support team for further assistance. | yes |
| t0005 | can you check what the problem with my account is, it seems to b e blocked | account_access | account_access | I cannot view specific reasons for blocked accounts, but I will pass your request to the team for review. They will contact you with the correct information. | yes |
| t0006 | is there a problem with my account, it appears to be blocked | account_access | account_access | I cannot see the specific reason for a blocked account status. Please contact our support team so they can review your account status and assist you. | yes |
| t0007 | did someone who is unauthorized try to get into my bank account | account_access | account_access | I cannot access your account details or reasons for unauthorized access to your account. Please contact our support team so they can assist you with this. | yes |
| t0008 | what have i spent things on | balance_or_statement | balance_or_statement | I cannot view your specific spending details here. Please log in to your online banking or mobile app to see your spending history. | yes |
| t0009 | is my visa bill over my limit this month | balance_or_statement | loans_and_credit | I cannot view your specific account details here. Please log in to your online banking or mobile app to view your current credit limit. | yes |
| t0010 | tell me how much i have been buying as of late | balance_or_statement | balance_or_statement | I cannot see your transaction history directly. Please log in to your online banking or mobile app to check your available balance. | yes |
| t0011 | check my wells fargo account to see if i have enough for these nike's | balance_or_statement | balance_or_statement | I cannot view account balances directly. Please log in to your online banking or mobile app to see your available funds. | yes |
| t0012 | what've i spent on gas this month | balance_or_statement | balance_or_statement | I cannot view your specific gas spending from this chat. Please log in to your online banking or mobile app to review your monthly gas transactions. | yes |
| t0013 | have i gone over my entertainment budget | balance_or_statement | other | I cannot provide feedback onwellowing your entertainment budget. Your request will be passed to a team member who can help. | no: Contains typo 'onwellowing' |
| t0014 | Is there a trick to get the disposable virtual card to work? | card_not_working | card_not_working | We are sorry for the confusion regarding your virtual card. Our team will review the status and advise you on the next steps. | yes |
| t0015 | My contactless is not working | card_not_working | card_not_working | I am sorry to hear about the issue with your contactless card. Please let our team know so they can assist you further. | yes |
| t0016 | I was getting cash and can't get my card back. | card_not_working | card_not_working | If you are unable to withdraw cash, please reach out to our customer service team for assistance. They will guide you on how to proceed with a replacement. | no: Customer reported card not returned, not cash withdrawal issue; implies action taken |
| t0017 | Is there a way to know when my card will arrive? | card_not_working | card_not_working | I cannot check the status of your delivery from here. Please contact our support team to get the latest delivery details. | yes |
| t0018 | i was trying to use my card to get food at publix but it got declined | card_not_working | card_not_working | I am sorry to hear your card was declined. Please contact our support team so they can investigate this for you. | yes |
| t0019 | why my card stop working | card_not_working | card_not_working | I am unable to see specific account details or reasons for blocked cards. Please contact our support team so they can assist you further. | yes |
| t0020 | my dog swallowed my credit card and i need a new one | lost_or_stolen_card | card_not_working | Please contact our support team directly so they can check your card status and assist you. | yes |
| t0021 | A fee was charged for my recent withdrawal. | fees_and_charges | fees_and_charges | I am unable to view your transaction history directly. Please contact our support team so they can review your account details. | yes |
| t0022 | I got a fee for an ATM withdrawal. | fees_and_charges | fees_and_charges | ATM withdrawal fees depend on the machine and your account type. Our team will review your account details to explain the charge. | yes |
| t0023 | I thought transfers were free, why was I charged a fee? | fees_and_charges | fees_and_charges | Transfer fees are applied based on the method used and your account type. We will pass your concern to the relevant team for clarification. | yes |
| t0024 | Are there extra charges for exchanging currency? | fees_and_charges | fees_and_charges | Currency exchange fees depend on the specific transaction type and the currency involved. Please share your account details so I can pass this to our team for review. | yes |
| t0025 | when i go to cuba will there be a transaction fee for using my discover card | fees_and_charges | fees_and_charges | Fees for using a Discover card depend on your specific card product and location. Our team will review your specific case and get back to you. | yes |
| t0026 | what's my checking account yielding in interest | fees_and_charges | fees_and_charges | I cannot view your account balances directly. Please log in to your online banking or mobile app to check your current interest. | yes |
| t0027 | what does my checking account at chase earn in interest | fees_and_charges | balance_or_statement | I do not have access to view personal account balances or Chase's account balances. Please log in to your online banking or mobile app to check your current balance. | yes |
| t0028 | look up the lowest amount i can pay on my electricity bill | loans_and_credit | bill_inquiry | Please log in to your account to view the latest amount. Our team will review your account details if you contact them. | yes |
| t0029 | could i perhaps increase by credit limit by 500 dollars | loans_and_credit | loans_and_credit | I cannot process credit limit increases directly. I will forward your request to the credit team to discuss this. | yes |
| t0030 | what's my spending limit on my discover | loans_and_credit | loans_and_credit | I cannot view your specific account limits directly. Please log in to your online banking or mobile app to view your current credit limit. | yes |
| t0031 | has my application for the hsbc card been approved yet | loans_and_credit | loans_and_credit | I do not have access to check application statuses or hsbc card application updates. Our team will review your request and contact you regarding the next steps. | yes |
| t0032 | My bags were stolen. I need a new card but need to cancel the stolen one. | lost_or_stolen_card | lost_or_stolen_card | You can block your card instantly through the mobile app or online banking. We will pass your request to the team to process this replacement. | yes |
| t0033 | I cant find my card, it's gone. | lost_or_stolen_card | lost_or_stolen_card | Please contact our support team immediately to report the loss. They will guide you on how to secure your account. | yes |
| t0034 | I cannot find my credit card. | lost_or_stolen_card | lost_or_stolen_card | Please contact our support team immediately to report your lost card. They will guide you on the next steps. | yes |
| t0035 | How do I report my card stolen? | lost_or_stolen_card | lost_or_stolen_card | To report your stolen card, please contact our customer service team directly. They will guide you through the reporting process. | yes |
| t0036 | please tell me how to go about reporting a lost card | lost_or_stolen_card | lost_or_stolen_card | Please contact our support team directly so they can guide you through reporting a lost card. | yes |
| t0037 | set up a report for a lost card | lost_or_stolen_card | lost_or_stolen_card | Please contact our customer service team immediately to report the lost card. They will guide you on how to request a replacement. | yes |
| t0038 | i need to report that someone ran off with my td bank card earlier | lost_or_stolen_card | card_not_working | I cannot process requested card issues directly. Please contact our support team to report this issue for further assistance. | yes |
| t0039 | can you monitor my package's progress | order_status | order_status | I cannot see your account details directly. Please contact our customer service team to get your current track. They will provide you with the latest details. | yes |
| t0040 | can you help with delivery tracking | order_status | order_status | I cannot access tracking details directly. Please contact our support team to get this information. | yes |
| t0041 | track my order | order_status | order_status | Please provide your order number so I can pass this to the team for investigation. | yes |
| t0042 | what is the status of the order i placed with you yesterday | order_status | order_status | I cannot access your specific order details directly. Please contact our support team to get the latest updates. | yes |
| t0043 | has my package been sent | order_status | order_status | I cannot see your specific order details directly. Please contact our support team to get the latest update for your package. | yes |
| t0044 | i need a delivery tracked | order_status | order_status | I will forward your inquiry about the delivery status to the logistics team for you. | yes |
| t0045 | What are the most current exchange rates? | other | other | I will pass your question about exchange rates to a specialist who can assist you further. | yes |
| t0046 | what are exchange rates | other | other | I will pass your question about exchange rates to a specialist who can assist you further. | yes |
| t0047 | ai is a helpful item to have in my home, thank you | other | missing_or_wrong_item | We apologize for the mix-up. Please share your order number so we can pass this to our team. | no: Customer did not report an issue or mix-up; the reply incorrectly assumes a problem occurred. |
| t0048 | enjoy your day, goodbye | other | other | Goodbye! Please reach out again if you need any assistance in the future. | yes |
| t0049 | My transfer did not go through. | transfer_issue | transfer_issue | If your transfer did not go through, it is usually blocked by the receiving bank. Please contact support so they can assist you further. | yes |
| t0050 | Why hasn't my transfer gone through yet? | transfer_issue | transfer_issue | I am sorry to hear about the delay with your transfer. Please share your transaction details so I can pass this to our team for review. | yes |
| t0051 | I am trying to make a transfer and am unsuccessful, can you please tell me why? | transfer_issue | transfer_issue | I am sorry to hear about the issue with your transfer. I will pass your query to the team to investigate why it is being declined. | yes |
| t0052 | Please explain why the transfers shows up as pending? | transfer_issue | transfer_issue | Pending transfers can sometimes take time to clear depending on the banks involved. We will pass your query to the team to check the status. | yes |
| t0053 | Why didn't my balance change after I transferred some money? | transfer_issue | transfer_issue | I am sorry to hear about the issue with your transfer. Please share the transaction details so I can pass this to our team for review. | yes |
| t0054 | You keep declining my transfers? It's always been working really well so far, but when I tried to buy something just now the card got declined. I tried couple times already but same thing. | transfer_issue | transfer_issue | Declines can happen due to security checks or insufficient funds. Please share the transaction details so we can pass this to our team for review. | yes |
| t0055 | transaction failed? | transfer_issue | transfer_issue | We are sorry for the failed transaction. Please share your reference number so we can pass this to our team. | yes |
| t0056 | Why was I charged multiple times for the same thing? | unrecognized_transaction | unrecognized_transaction | Duplicate charges can sometimes be due to temporary security checks or temporary holds. Please contact our support team so they can review your account details. | yes |
| t0057 | im so mad right now. theres several charges that I think my x boyfriend made on my card. the companys on the website wouldn't refund me my money, they told me to contact my bank. DO something please. | unrecognized_transaction | refund_request | We are sorry for the confusion. Please share your transaction details so we can pass your request to the team. | yes |
| t0058 | Hello. I checked my transactions and saw I was double charged on my card for a restaurant visit. I would like to get my money back. | unrecognized_transaction | unrecognized_transaction | Please contact our fraud team immediately to report this transaction. They will guide you on the next steps to secure your account. | yes |
| t0059 | There is a payment that is not mine in the app. Please advise/ | unrecognized_transaction | unrecognized_transaction | Please report this unexpected payment to our fraud team immediately. They will review your account and advise on the next steps. | yes |
| t0060 | my online bill has some items that i didn't purchase from penneys | unrecognized_transaction | card_not_working | I am sorry to hear about the issue with your online bill. Please let me know which items you need so I can pass this to the team. | yes |
| t0061 | this charge is bs | unrecognized_transaction | unrecognized_transaction | Please reach out to our customer service team to report this bs and get back to you. | no: Reply repeats the customer's profanity 'bs' which is impolite. |
| t0062 | i think someone jacked my bank account | unrecognized_transaction | account_access | I am sorry to hear about the trouble with your account. Please contact our support team so they can assist you with the next steps. | yes |
| t0063 | ok good chatting goodbye | other | other | You are very welcome. Please let me know if there is anything else I can help you with. | no: Reply ignores the customer's explicit goodbye and attempts to continue the conversation. |
| t0064 | what is albert einstein famous for | other | other | I am not able to provide information about the history of works. Your request will be passed on to someone who can help. | yes |

</details>
