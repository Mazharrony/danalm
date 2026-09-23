# Phase 2 results: data

Generated on 2026-09-23 by `scripts/phase2_report.py` from the run outputs; do not edit by hand. Decisions: D-019 to D-022 in [DECISIONS.md](../DECISIONS.md). No native speaker has reviewed the synthetic data yet.

## Pretraining corpus

Tokenizer danalm-v1 (sha256 `be583dda7da5…`). Tokens are text tokens; the shards add one EOS per document.

| Source | Licence | Docs (raw → kept) | Characters | Tokens | Share | Chars/token |
|---|---|---:|---:|---:|---:|---:|
| fineweb2-arb | ODC-By-1.0 (+ Common Crawl ToU) | 696,581 → 696,566 | 1,963,092,610 | 550,372,592 | 36.8% | 3.57 |
| fineweb-edu | ODC-By-1.0 (+ Common Crawl ToU) | 391,528 → 390,781 | 1,841,143,384 | 480,273,959 | 32.2% | 3.83 |
| fineweb2-ars | ODC-By-1.0 (+ Common Crawl ToU) | 211,929 → 211,926 | 934,208,621 | 299,889,655 | 20.1% | 3.12 |
| wikipedia-ar | CC-BY-SA-3.0 + GFDL | 287,818 → 287,667 | 343,959,319 | 100,343,674 | 6.7% | 3.43 |
| wikipedia-en | CC-BY-SA-3.0 + GFDL | 59,362 → 59,338 | 169,464,504 | 49,962,549 | 3.3% | 3.39 |
| bitext-banking | CDLA-Sharing-1.0 | 25,545 → 25,545 | 24,525,088 | 5,401,059 | 0.4% | 4.54 |
| bitext-support | CDLA-Sharing-1.0 | 26,872 → 26,872 | 18,304,872 | 3,937,071 | 0.3% | 4.65 |
| bitext-telco | CDLA-Sharing-1.0 | 26,000 → 25,961 | 14,068,652 | 3,305,755 | 0.2% | 4.26 |
| clinc-train | CC-BY-3.0 | 15,250 → 15,120 | 607,144 | 164,838 | <0.1% | 3.68 |
| massive-ar-train | CC-BY-4.0 | 11,514 → 10,403 | 299,529 | 87,184 | <0.1% | 3.44 |
| **Total** | | 1,752,399 → 1,750,179 | 5,309,673,723 | 1,493,738,336 | 100% | 3.55 |

| Split | Tokens (with EOS) | Shards |
|---|---:|---:|
| train | 1,487,850,948 | 15 |
| val | 7,637,567 | 1 |

**Cleaning** (`scripts/prepare_data.py`): exact dup 1,174, too short 921, low letter ratio 123, repetitive 2 dropped, of 1,752,399 documents read.

**Language tags** (heuristic, D-011; the Arabizi tag has known false positives on English text such as model numbers): ar 1,169,694, en 527,901, mixed 36,557, arabizi 16,027.

**Shard check** (`scripts/verify_shards.py`): train OK (1,741,445 EOS for 1,741,445 docs, max id 16383 < 16384, 414 docs decoded exactly of which 14 cross a shard boundary); val OK (8,734 EOS for 8,734 docs, max id 16383 < 16384, 400 docs decoded exactly of which 0 cross a shard boundary).
## Synthetic SFT data

### Generation

Teacher: Qwen3.5-35B-A3B UD-Q4_K_XL (unsloth/Qwen3.5-35B-A3B-GGUF) (revision `bc014a17`), 3,801 requests of 8 examples each. Answered: 3,801 (failed after retries: 0; reused from a crashed attempt: 0). 1,509,660 teacher tokens, 4.6 h, 92 tok/s.

| Variety | Parsed | Kept by the filters | Rejected (reason: count) |
|---|---:|---:|---|
| gulf_arabic | 6,315 | 5,606 (88.8%) | non_gulf_dialect 332, reply_claims_action 217, bad_fields 58, message_lang_mixed 43, msa_in_message 41, duplicate 17, reply_lang_mixed 1 |
| english | 6,041 | 5,807 (96.1%) | reply_claims_action 181, bad_fields 40, duplicate 7, near_duplicate 6 |
| arabizi | 7,534 | 6,080 (80.7%) | duplicate 478, reply_lang_en 371, message_lang_ar 352, bad_fields 143, near_duplicate 53, message_lang_mixed 51, reply_lang_mixed 4, reply_lang_ar 2 |
| mixed | 10,398 | 3,392 (32.6%) | message_lang_ar 4,380, message_lang_en 1,352, non_gulf_dialect 332, msa_in_message 302, reply_claims_action 237, reply_lang_en 217, bad_fields 162, near_duplicate 13, duplicate 11 |
| **all** | 30,288 | 20,885 (69.0%) | |

### Judge (blind re-labelling by the same model)

The judge sees only the message and the 21 intent descriptions. Rows where it disagrees with the intended label are dropped from the final set.

| Variety | Kept rows | Judge agrees | No valid label |
|---|---:|---:|---:|
| gulf_arabic | 5,606 | 5,132 (91.5%) | 1 |
| english | 5,807 | 5,348 (92.1%) | 2 |
| arabizi | 6,080 | 4,944 (81.3%) | 6 |
| mixed | 3,392 | 2,809 (82.8%) | 6 |
| **all** | 20,885 | 18,233 (87.3%) | 15 |

Agreement per intent, lowest first:

| Intent | Rows | Agrees |
|---|---:|---:|
| other | 904 | 51.9% |
| failed_delivery | 1,057 | 76.1% |
| roaming | 1,092 | 80.4% |
| refund_request | 1,028 | 81.0% |
| cancel_order | 918 | 81.2% |
| order_status | 1,022 | 81.7% |
| lost_or_stolen_card | 957 | 84.6% |
| bill_inquiry | 1,042 | 85.0% |
| change_delivery_details | 965 | 86.5% |
| missing_or_wrong_item | 1,047 | 87.9% |
| balance_or_statement | 1,083 | 90.6% |
| transfer_issue | 972 | 90.7% |
| loans_and_credit | 1,061 | 90.9% |
| unrecognized_transaction | 958 | 91.5% |
| card_not_working | 986 | 93.8% |
| fees_and_charges | 1,091 | 94.1% |
| plan_change | 927 | 94.6% |
| handoff_to_human | 922 | 96.1% |
| sim_or_number | 1,018 | 96.5% |
| account_access | 914 | 98.9% |
| network_or_internet_issue | 921 | 99.1% |

Most frequent disagreements (intended → judge):

- other → handoff_to_human: 225
- failed_delivery → order_status: 213
- order_status → failed_delivery: 172
- cancel_order → order_status: 126
- lost_or_stolen_card → card_not_working: 106
- refund_request → failed_delivery: 91
- change_delivery_details → order_status: 66
- other → account_access: 65
- missing_or_wrong_item → order_status: 60
- roaming → plan_change: 59

### Final SFT set

18,233 examples (train 17,286, val 947), 1,132,278 tokens (message + JSON target, danalm-v1).

| Variety | Examples | Avg words per message | Distinct word bigrams (messages) |
|---|---:|---:|---:|
| gulf_arabic | 5,132 | 9.3 | 0.52 |
| english | 5,348 | 11.0 | 0.41 |
| arabizi | 4,944 | 7.9 | 0.60 |
| mixed | 2,809 | 9.8 | 0.55 |

Examples per intent: min 469 (other), max 1,027 (fees_and_charges). Smallest intent × variety cells: other/mixed 13, unrecognized_transaction/mixed 69, plan_change/mixed 69, network_or_internet_issue/mixed 81, handoff_to_human/mixed 81.

Random training examples (seed 7):

| Variety | Intent | Message | Reply |
|---|---|---|---|
| gulf_arabic | roaming | يا هلا، شو باجرو الرومينغ لانه ابغي اسافر وغايب عنه وايد، وين اشوفه؟ | تفضل توجه للتطبيق واختار الباقات المتاحة، وسنقوم بمتابعة طلبك ونخبرك بالتفاصيل. |
| gulf_arabic | fees_and_charges | شو هالفية اللي حطيتوا عليها لبطاقتي؟ ابغي اعرف. | مرحبا، يمكن يكون رسوم تحويل أو صيانة. نرجو الاتصال بنا لنوضح لكم التفاصيل. |
| gulf_arabic | missing_or_wrong_item | الوجبة تبغى تبرد شو سويت؟ ابغيها ساخنة. | نعتذر من التبريد، نرسل طلبك للفريق ليحل المشكلة معك. |
| gulf_arabic | lost_or_stolen_card | شنو الوان الحين ما قدرت اجد الكرت عقب ما اخريته | نوصيك بطلب استبدال عبر التطبيق وسيتم تحويل الطلب للفريق المختص للتحقق من الوضع. |
| english | balance_or_statement | need bank statement from january, can u email it to me right now? | I am unable to process email requests or access your data. You should try downloading it via the official website or app. |
| english | failed_delivery | Courier failed to deliver again, u sent wrong address? Very annoying. | Please share your order number so we can forward this to our team. |
| english | balance_or_statement | Need transaction history for last month asap, thx. | Your transaction history is available on the official website. You may also request a statement from our customer service. |
| english | roaming | Pls help! My phone shows no signal while in Dubai airport, thinking about roaming. | If you are inside the UAE, you do not need international roaming. Please ensure mobile data is turned on in settings. |
| arabizi | balance_or_statement | wain li statement shhar jaye? 3indi 3ashshak kthir | t7eb tiftah app w t9ra min hna, aw nba'atho lil team y3awdok. |
| arabizi | account_access | laish ma jaani otp? mobi zain. | yalla jreb tu3id kalimat al-7abib. idha laish, nba3thak lil-far' yshofhu. |
| arabizi | bill_inquiry | shlon al-bill? laish 3am y3id? | ahlan, li tshuf al-tafseel min app. ila hayej shu 3ala abgha n7awl atb2tho lalfireeq. |
| arabizi | lost_or_stolen_card | karti zala, abi a7sahha laish ma3rfa shu. | zain, abgha tba3ath 2al3umana w nkhdm lik l3aradh ya3mil lik al7saha. |
| mixed | balance_or_statement | هل يمكنكم تزويدي ب statement لشهر مارس؟ | يمكنك تحميل ال statement من قسم الخدمات، وسأبلغ الفريق لمتابعة طلبك. |
| mixed | missing_or_wrong_item | موجود في التطبيق delivered بس الكيس مفتوح and the food spilled. | ندرس حالتك الآن ونرسلها للفريق المختص للمعالجة. |
| mixed | missing_or_wrong_item | ال driver وضع الطلب في مكان خاطئ وفيه damage، شو الحل؟ | نأسف لهذه التجربة، سيقوم الفريق بمراجعة الحالة والاتصال بك قريبا. |
| mixed | balance_or_statement | توقعته يوصلني statement بالبريد بس ما وصل حتى الآن. | الرسائل الإلكترونية يتم إرسالها فورا، يرجى التحقق من مجلد البريد غير المرغوب فيه. |
