# Phase 6 results: evaluation on the human test set (English part)

Generated on 2026-09-25 by `scripts/phase6_report.py`; do not edit by hand. Protocol: D-032, bars: D-030 in [DECISIONS.md](../DECISIONS.md).

**Test set:** 64 English messages over 10 intents, labelled by the owner (MR), SHA-256 `e6961fd244c5…`. The overlap check with all training data reports 0 overlaps. The Gulf Arabic, Arabizi and mixed parts of the test set do not exist yet, so **this evaluation covers English only**. With 64 messages, one message is 1.6 points, so the intervals are wide.

## Results

| System | Intent accuracy (95% interval) | Macro-F1 | Valid JSON | Reply in the right language |
|---|---|---:|---|---|
| DanaLM (62M, Phase 5 model) | 56.2% (36/64; 44–68%) | 57.8% | 100.0% (64/64; 94–100%) | 100.0% (64/64; 94–100%) |
| CAMeLBERT-mix classifier (110M) | 46.9% (30/64; 35–59%) | 46.4% | – | – |
| Qwen3.5-35B-A3B, zero-shot | 89.1% (57/64; 79–95%) | 90.3% | – | – |

- Messages without a usable intent, counted as wrong (D-032): DanaLM (62M, Phase 5 model) 0; CAMeLBERT-mix classifier (110M) 0; Qwen3.5-35B-A3B, zero-shot 0. (DanaLM: an answer that is not valid JSON; Qwen: a message its answer lines did not label.)
- Macro-F1 difference, DanaLM minus CAMeLBERT-mix classifier (110M): +11.5 points (paired bootstrap 95% interval +1.0 to +22.0).
- Macro-F1 difference, DanaLM minus Qwen3.5-35B-A3B, zero-shot: -32.4 points (paired bootstrap 95% interval -46.7 to -20.8).

## The D-030 bars (English part only)

| Bar | Target | Measured | Result |
|---|---|---|---|
| Valid JSON | ≥ 99% | 100.0% (64/64; 94–100%) | **met** |
| Intent macro-F1 vs CAMeLBERT | within 3 points | +11.5 points (95% interval +1.0 to +22.0) | **met** |
| Coverage at the SFT-validation threshold (0.612) | ≥ 70% answered, ≥ 95% of them right | 85.9% answered (55/64), 61.8% right | **missed** |
| Reply in the right language | ≥ 95% | 100.0% (64/64; 94–100%) | **met** |

## Reply quality

Qwen judged the 64 valid DanaLM replies it gave a full verdict for on four yes/no questions: answers 93.8% (60/64; 85–98%), polite clear 98.4% (63/64; 92–100%), language 100.0% (64/64; 94–100%), safe 96.9% (62/64; 89–99%). **All four yes: 89.1% (57/64; 79–95%).** Qwen wrote DanaLM's training replies, so it may be lenient; every reply is listed at the end for the owner's spot check.

## Per intent

| Intent | Messages | DanaLM (62M, Phase 5 model) | CAMeLBERT-mix classifier (110M) | Qwen3.5-35B-A3B, zero-shot |
|---|---:|---:|---:|---:|
| lost_or_stolen_card | 8 | 4/8 | 3/8 | 8/8 |
| account_access | 7 | 3/7 | 4/7 | 6/7 |
| fees_and_charges | 7 | 5/7 | 5/7 | 5/7 |
| transfer_issue | 7 | 6/7 | 4/7 | 6/7 |
| unrecognized_transaction | 7 | 1/7 | 0/7 | 6/7 |
| balance_or_statement | 6 | 1/6 | 0/6 | 6/6 |
| card_not_working | 6 | 4/6 | 4/6 | 5/6 |
| order_status | 6 | 4/6 | 3/6 | 6/6 |
| other | 6 | 6/6 | 6/6 | 6/6 |
| loans_and_credit | 4 | 2/4 | 1/4 | 3/4 |

## Sensitivity: without the 3 disputed messages

An independent audit disagreed with the owner's label on review rows 28, 57, 58. Leaving those messages out (the owner's labels stay the primary result): DanaLM (62M, Phase 5 model) 55.7% (34/61; 43–67%); CAMeLBERT-mix classifier (110M) 47.5% (29/61; 36–60%); Qwen3.5-35B-A3B, zero-shot 91.8% (56/61; 82–96%).

## Every DanaLM error (28 of 64)

| Id | Message | Gold | DanaLM intent | Confidence | DanaLM reply |
|---|---|---|---|---:|---|
| t0002 | I need a new passcode. | account_access | other | 0.87 | You are welcome! Please let me know if you need anything later. |
| t0005 | can you check what the problem with my account is, it seems to b e blocked | account_access | other | 0.16 | I am unable to access your account details here. I will pass your request to a specialist who can help. |
| t0006 | is there a problem with my account, it appears to be blocked | account_access | card_not_working | 0.69 | We will pass your details to the relevant team to investigate this issue. |
| t0007 | did someone who is unauthorized try to get into my bank account | account_access | other | 0.38 | I am unable to provide banking advice or try banking services. I will pass your question to someone who can help. |
| t0008 | what have i spent things on | balance_or_statement | other | 1.00 | I do not have access to current stock market data. I will forward your question to the appropriate team for assistance. |
| t0009 | is my visa bill over my limit this month | balance_or_statement | bill_inquiry | 0.99 | I cannot view your specific balance here. Please reach out to our support team for a review. |
| t0011 | check my wells fargo account to see if i have enough for these nike's | balance_or_statement | account_access | 0.44 | Please try using the mobile app instead of the website. If the problem continues, we will pass your details to the team. |
| t0012 | what've i spent on gas this month | balance_or_statement | other | 1.00 | I do not have access to your personal account information. I will forward your question to the appropriate support team. |
| t0013 | have i gone over my entertainment budget | balance_or_statement | other | 1.00 | I am here to assist with customer service inquiries and account support. I will forward your question about gone-bourging to the appropriate support team. |
| t0014 | Is there a trick to get the disposable virtual card to work? | card_not_working | fees_and_charges | 0.64 | We cannot see your account details here. Please visit a branch or call us to get the correct information. |
| t0018 | i was trying to use my card to get food at publix but it got declined | card_not_working | unrecognized_transaction | 0.93 | We cannot see the transaction status here. Please call our hotline so they can investigate this. |
| t0020 | my dog swallowed my credit card and i need a new one | lost_or_stolen_card | loans_and_credit | 0.55 | You can start the process through our mobile app. We will forward your request to the relevant department. |
| t0026 | what's my checking account yielding in interest | fees_and_charges | other | 0.91 | I am unable to provide current rates, but I will pass your query to the relevant team. |
| t0027 | what does my checking account at chase earn in interest | fees_and_charges | other | 0.98 | I am here to assist with banking services. I will pass your question about chase earn to someone who can help. |
| t0028 | look up the lowest amount i can pay on my electricity bill | loans_and_credit | bill_inquiry | 1.00 | You can try using a different payment method online or by calling our support line. We will pass your concern to the billing team. |
| t0030 | what's my spending limit on my discover | loans_and_credit | balance_or_statement | 0.49 | I am unable to access your personal balance. Please log in to your account to view your current balance. |
| t0034 | I cannot find my credit card. | lost_or_stolen_card | loans_and_credit | 0.99 | Please contact our support team directly so they can assist you. |
| t0037 | set up a report for a lost card | lost_or_stolen_card | fees_and_charges | 0.42 | I cannot process this request. Please submit a request so the team can review it. |
| t0038 | i need to report that someone ran off with my td bank card earlier | lost_or_stolen_card | card_not_working | 0.73 | We cannot view your account details here. Please contact our support team directly for help. |
| t0039 | can you monitor my package's progress | order_status | other | 0.99 | I do not have access to current news updates. I will forward your inquiry to the relevant team. |
| t0040 | can you help with delivery tracking | order_status | other | 0.78 | I am unable to provide details on tracking locations. I will pass your request to a specialist who can help. |
| t0053 | Why didn't my balance change after I transferred some money? | transfer_issue | balance_or_statement | 0.97 | I am unable to access your balance directly. Please log in to your account to view the current balance. |
| t0056 | Why was I charged multiple times for the same thing? | unrecognized_transaction | bill_inquiry | 0.72 | We do not have access to your account details here. Please contact support for a detailed review. |
| t0057 | im so mad right now. theres several charges that I think my x boyfriend made on my card. the companys on the website wouldn't refund me my money, they told me to contact my bank. DO something please. | unrecognized_transaction | refund_request | 0.99 | We are sorry for the issue. Please provide your order details so we can pass your request to the team. |
| t0058 | Hello. I checked my transactions and saw I was double charged on my card for a restaurant visit. I would like to get my money back. | unrecognized_transaction | refund_request | 0.63 | We are sorry for the confusion. Please provide your order details so we can pass this to our team. |
| t0060 | my online bill has some items that i didn't purchase from penneys | unrecognized_transaction | bill_inquiry | 1.00 | I am unable to process changes directly. Please reach out to our support team who can guide you through the process. |
| t0061 | this charge is bs | unrecognized_transaction | bill_inquiry | 0.50 | We cannot process payments directly here. Please visit a branch or call us for assistance. |
| t0062 | i think someone jacked my bank account | unrecognized_transaction | other | 0.96 | I am here to assist with banking services. I will pass your question about bank hours to a team member who can help. |

## Latency and memory

Per message, batch 1, greedy decoding with the current decoder (no KV cache yet; Phase 7 adds it and quantizes). Median and p95 over the 64 messages after warm-up. CPU: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 4 threads.

| | CPU, float32 | GPU (RTX 4070), bf16 |
|---|---:|---:|
| Median per message | 0.63 s | 0.37 s |
| p95 per message | 0.92 s | 0.43 s |
| Answer tokens (mean) | 39 | 39 |
| Tokens per second | 60 | 105 |
| Memory | weights 237 MB; process peak 1203 MB (before loading the model: 926 MB) | peak allocated 370 MB |

## All replies (for the owner's spot check)

<details><summary>Show the 64 answers</summary>

| Id | Message | Gold | DanaLM | Reply | Qwen: good? |
|---|---|---|---|---|---|
| t0001 | I do not know my passcode. | account_access | account_access | You can request a new one by logging into your account. We will pass this to the technical team. | no: Claims 'We will pass this' which implies the request is already submitted, but the customer hasn't requested it yet; also implies the bot can pass specific passcode requests which it cannot do directly. |
| t0002 | I need a new passcode. | account_access | other | You are welcome! Please let me know if you need anything later. | no: Does not answer the customer's need for a new passcode; ignores the request entirely. |
| t0003 | Can I be given a new passcode? | account_access | account_access | You can request a new one by calling our support line. We will pass your details to the team. | yes |
| t0004 | I lost my password | account_access | account_access | Please try resetting your password via the link on the screen. | yes |
| t0005 | can you check what the problem with my account is, it seems to b e blocked | account_access | other | I am unable to access your account details here. I will pass your request to a specialist who can help. | yes |
| t0006 | is there a problem with my account, it appears to be blocked | account_access | card_not_working | We will pass your details to the relevant team to investigate this issue. | yes |
| t0007 | did someone who is unauthorized try to get into my bank account | account_access | other | I am unable to provide banking advice or try banking services. I will pass your question to someone who can help. | yes |
| t0008 | what have i spent things on | balance_or_statement | other | I do not have access to current stock market data. I will forward your question to the appropriate team for assistance. | yes |
| t0009 | is my visa bill over my limit this month | balance_or_statement | bill_inquiry | I cannot view your specific balance here. Please reach out to our support team for a review. | yes |
| t0010 | tell me how much i have been buying as of late | balance_or_statement | balance_or_statement | I am unable to check your current balance. Please log in to your account to view your current balance. | yes |
| t0011 | check my wells fargo account to see if i have enough for these nike's | balance_or_statement | account_access | Please try using the mobile app instead of the website. If the problem continues, we will pass your details to the team. | yes |
| t0012 | what've i spent on gas this month | balance_or_statement | other | I do not have access to your personal account information. I will forward your question to the appropriate support team. | yes |
| t0013 | have i gone over my entertainment budget | balance_or_statement | other | I am here to assist with customer service inquiries and account support. I will forward your question about gone-bourging to the appropriate support team. | no: Contains typo 'gone-bourging' and is unclear |
| t0014 | Is there a trick to get the disposable virtual card to work? | card_not_working | fees_and_charges | We cannot see your account details here. Please visit a branch or call us to get the correct information. | yes |
| t0015 | My contactless is not working | card_not_working | card_not_working | We will pass your details to the team to check the status. | yes |
| t0016 | I was getting cash and can't get my card back. | card_not_working | card_not_working | Please try a different payment method or contact our support team for help. | yes |
| t0017 | Is there a way to know when my card will arrive? | card_not_working | card_not_working | We cannot see your specific card status here. Please reach out to our support team for assistance. | yes |
| t0018 | i was trying to use my card to get food at publix but it got declined | card_not_working | unrecognized_transaction | We cannot see the transaction status here. Please call our hotline so they can investigate this. | yes |
| t0019 | why my card stop working | card_not_working | card_not_working | Please try using a different payment method first. Our team will review your case. | yes |
| t0020 | my dog swallowed my credit card and i need a new one | lost_or_stolen_card | loans_and_credit | You can start the process through our mobile app. We will forward your request to the relevant department. | yes |
| t0021 | A fee was charged for my recent withdrawal. | fees_and_charges | fees_and_charges | Our system shows the transaction was declined by the team. We will pass this to them. | no: Claims the system shows the transaction was declined by the team, which invents a fact and implies the bot has access to account status. |
| t0022 | I got a fee for an ATM withdrawal. | fees_and_charges | fees_and_charges | We will pass your concern to the team to investigate this. | yes |
| t0023 | I thought transfers were free, why was I charged a fee? | fees_and_charges | fees_and_charges | I cannot see your account details. Please contact our team to review your case. | yes |
| t0024 | Are there extra charges for exchanging currency? | fees_and_charges | fees_and_charges | Please check the policy on our official site. Our team will review your case and get back to you. | yes |
| t0025 | when i go to cuba will there be a transaction fee for using my discover card | fees_and_charges | fees_and_charges | We will pass your query to the relevant team to check your specific plan details. | yes |
| t0026 | what's my checking account yielding in interest | fees_and_charges | other | I am unable to provide current rates, but I will pass your query to the relevant team. | yes |
| t0027 | what does my checking account at chase earn in interest | fees_and_charges | other | I am here to assist with banking services. I will pass your question about chase earn to someone who can help. | yes |
| t0028 | look up the lowest amount i can pay on my electricity bill | loans_and_credit | bill_inquiry | You can try using a different payment method online or by calling our support line. We will pass your concern to the billing team. | yes |
| t0029 | could i perhaps increase by credit limit by 500 dollars | loans_and_credit | loans_and_credit | I am unable to change limits directly. Please submit a request via the official app and we will review it. | yes |
| t0030 | what's my spending limit on my discover | loans_and_credit | balance_or_statement | I am unable to access your personal balance. Please log in to your account to view your current balance. | yes |
| t0031 | has my application for the hsbc card been approved yet | loans_and_credit | loans_and_credit | I do not have access to your specific case. Please submit a request via the app. | yes |
| t0032 | My bags were stolen. I need a new card but need to cancel the stolen one. | lost_or_stolen_card | lost_or_stolen_card | Please reach out to our customer service team right away so they can assist you with blocking. | yes |
| t0033 | I cant find my card, it's gone. | lost_or_stolen_card | lost_or_stolen_card | Please reach out to our customer service team right away so they can assist you. | yes |
| t0034 | I cannot find my credit card. | lost_or_stolen_card | loans_and_credit | Please contact our support team directly so they can assist you. | yes |
| t0035 | How do I report my card stolen? | lost_or_stolen_card | lost_or_stolen_card | Contact our team via the official hotline immediately. | yes |
| t0036 | please tell me how to go about reporting a lost card | lost_or_stolen_card | lost_or_stolen_card | You must call our support line immediately to report the loss. | yes |
| t0037 | set up a report for a lost card | lost_or_stolen_card | fees_and_charges | I cannot process this request. Please submit a request so the team can review it. | yes |
| t0038 | i need to report that someone ran off with my td bank card earlier | lost_or_stolen_card | card_not_working | We cannot view your account details here. Please contact our support team directly for help. | yes |
| t0039 | can you monitor my package's progress | order_status | other | I do not have access to current news updates. I will forward your inquiry to the relevant team. | yes |
| t0040 | can you help with delivery tracking | order_status | other | I am unable to provide details on tracking locations. I will pass your request to a specialist who can help. | yes |
| t0041 | track my order | order_status | order_status | I cannot see your account details. Please contact our support team for an update. | yes |
| t0042 | what is the status of the order i placed with you yesterday | order_status | order_status | I am unable to access your specific order details. Please contact our support team for the latest details. | yes |
| t0043 | has my package been sent | order_status | order_status | I do not have access to live tracking data. Please submit a ticket so the team can investigate. | yes |
| t0044 | i need a delivery tracked | order_status | order_status | I am unable to access your specific order details. Please reach out to customer service for assistance. | yes |
| t0045 | What are the most current exchange rates? | other | other | I do not have access to current exchange rates. I will forward your question to the relevant team. | yes |
| t0046 | what are exchange rates | other | other | I do not have access to real-time exchange rates. I will forward your question to the relevant team. | yes |
| t0047 | ai is a helpful item to have in my home, thank you | other | other | You are welcome! Is there anything else I can help you with? | yes |
| t0048 | enjoy your day, goodbye | other | other | You are welcome! We are passing your message to the right person. | no: Reply is inappropriate for a goodbye message; it incorrectly states a message is being passed to a person when the customer is leaving. |
| t0049 | My transfer did not go through. | transfer_issue | transfer_issue | We are sorry for the delay. Your request will be passed to our team to investigate. | yes |
| t0050 | Why hasn't my transfer gone through yet? | transfer_issue | transfer_issue | We are sorry for the delay. We will forward your details to the support team to check. | yes |
| t0051 | I am trying to make a transfer and am unsuccessful, can you please tell me why? | transfer_issue | transfer_issue | We are sorry for the delay. Your case will be reviewed by our team shortly. | yes |
| t0052 | Please explain why the transfers shows up as pending? | transfer_issue | transfer_issue | We do not have access to your account details. Please reach out to our support team. | yes |
| t0053 | Why didn't my balance change after I transferred some money? | transfer_issue | balance_or_statement | I am unable to access your balance directly. Please log in to your account to view the current balance. | yes |
| t0054 | You keep declining my transfers? It's always been working really well so far, but when I tried to buy something just now the card got declined. I tried couple times already but same thing. | transfer_issue | transfer_issue | We are sorry for the trouble. We will pass your details to the team to check the status. | yes |
| t0055 | transaction failed? | transfer_issue | transfer_issue | We are escalating this to our specialists to investigate the issue. | yes |
| t0056 | Why was I charged multiple times for the same thing? | unrecognized_transaction | bill_inquiry | We do not have access to your account details here. Please contact support for a detailed review. | yes |
| t0057 | im so mad right now. theres several charges that I think my x boyfriend made on my card. the companys on the website wouldn't refund me my money, they told me to contact my bank. DO something please. | unrecognized_transaction | refund_request | We are sorry for the issue. Please provide your order details so we can pass your request to the team. | no: Customer asked for immediate action ('DO something'), but reply only asks for details without addressing the urgency or confirming escalation. |
| t0058 | Hello. I checked my transactions and saw I was double charged on my card for a restaurant visit. I would like to get my money back. | unrecognized_transaction | refund_request | We are sorry for the confusion. Please provide your order details so we can pass this to our team. | yes |
| t0059 | There is a payment that is not mine in the app. Please advise/ | unrecognized_transaction | unrecognized_transaction | We will forward your concern to the security team right away. | yes |
| t0060 | my online bill has some items that i didn't purchase from penneys | unrecognized_transaction | bill_inquiry | I am unable to process changes directly. Please reach out to our support team who can guide you through the process. | yes |
| t0061 | this charge is bs | unrecognized_transaction | bill_inquiry | We cannot process payments directly here. Please visit a branch or call us for assistance. | yes |
| t0062 | i think someone jacked my bank account | unrecognized_transaction | other | I am here to assist with banking services. I will pass your question about bank hours to a team member who can help. | no: Reply incorrectly claims the query is about bank hours instead of a security breach. |
| t0063 | ok good chatting goodbye | other | other | You are welcome! Is there anything else I can help you with? | yes |
| t0064 | what is albert einstein famous for | other | other | I do not have information on that subject. I will pass your query to someone who can help. | yes |

</details>
