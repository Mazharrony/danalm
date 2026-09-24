# Phase 2 results: data

Generated on 2026-09-24 by `scripts/phase2_report.py` from the run outputs; do not edit by hand. Decisions: D-019 to D-025 in [DECISIONS.md](../DECISIONS.md). No native speaker has reviewed the synthetic data yet.

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

### Generation (full run)

Teacher: Qwen3.5-35B-A3B UD-Q4_K_XL (unsloth/Qwen3.5-35B-A3B-GGUF) (revision `bc014a17`), 3,801 requests of 8 examples each. Answered: 3,801 (failed after retries: 0; reused from a crashed attempt: 0). 1,509,660 teacher tokens, 4.6 h, 92 tok/s.

| Variety | Parsed | Kept by the filters | Rejected (reason: count) |
|---|---:|---:|---|
| gulf_arabic | 6,315 | 5,606 (88.8%) | non_gulf_dialect 332, reply_claims_action 217, bad_fields 58, message_lang_mixed 43, msa_in_message 41, duplicate 17, reply_lang_mixed 1 |
| english | 6,041 | 5,807 (96.1%) | reply_claims_action 181, bad_fields 40, duplicate 7, near_duplicate 6 |
| arabizi | 7,534 | 6,080 (80.7%) | duplicate 478, reply_lang_en 371, message_lang_ar 352, bad_fields 143, near_duplicate 53, message_lang_mixed 51, reply_lang_mixed 4, reply_lang_ar 2 |
| mixed | 10,398 | 3,392 (32.6%) | message_lang_ar 4,380, message_lang_en 1,352, non_gulf_dialect 332, msa_in_message 302, reply_claims_action 237, reply_lang_en 217, bad_fields 162, near_duplicate 13, duplicate 11 |
| saudi_arabic | 0 | 0 (–) |  |
| **all** | 30,288 | 20,885 (69.0%) | |

### Second pass (D-023, D-024)

- **Arabizi replies in Gulf Arabic script (D-023):** 5,747 of 6,080 Arabizi messages got a reply that passed the checks (rejected: reply_claims_action 266, no_reply 62, non_gulf_dialect 5).
- **Real out-of-scope messages for `other` (D-024):** clinc-oos-train 250 of 15,250 rows, massive-ar-train-oos 250 of 11,514 rows; 491 got a reply that passed the checks.
- **Code-mixed `other` top-up:** 40 requests, 320 examples parsed, 96 kept (rejected: message_lang_ar 176, message_lang_en 34, non_gulf_dialect 7, reply_claims_action 5, duplicate 1, near_duplicate 1).

Merge before the single judge pass (the language tag is recomputed with the current tagger, and duplicates across sources are dropped):

| Source | Variety | Kept | Dropped (reason: count) |
|---|---|---:|---|
| sft-arabizi-replies | arabizi | 5,741 | message_lang_en 6 |
| sft-full | english | 5,807 | none |
| sft-full | gulf_arabic | 5,606 | none |
| sft-full | mixed | 3,392 | none |
| sft-other-mixed | mixed | 96 | none |
| sft-other-real | english | 248 | none |
| sft-other-real | saudi_arabic | 218 | message_lang_mixed 19, message_lang_en 6 |

### Judge (one blind re-labelling pass over all candidates)

The judge (the same model as the teacher) sees only the message and the 21 intent descriptions (the final ones, D-021). Rows where it disagrees with the intended label are dropped from the final set.

| Variety | Kept rows | Judge agrees | No valid label |
|---|---:|---:|---:|
| gulf_arabic | 5,606 | 5,152 (91.9%) | 0 |
| english | 6,055 | 5,614 (92.7%) | 0 |
| arabizi | 5,741 | 4,683 (81.6%) | 1 |
| mixed | 3,488 | 2,884 (82.7%) | 3 |
| saudi_arabic | 218 | 214 (98.2%) | 0 |
| **all** | 21,108 | 18,547 (87.9%) | 4 |

Agreement per intent, lowest first:

| Intent | Rows | Agrees |
|---|---:|---:|
| other | 1,462 | 68.7% |
| failed_delivery | 1,033 | 74.2% |
| missing_or_wrong_item | 1,042 | 82.1% |
| order_status | 1,012 | 82.6% |
| lost_or_stolen_card | 937 | 83.1% |
| roaming | 1,087 | 84.5% |
| refund_request | 995 | 84.7% |
| bill_inquiry | 1,030 | 85.6% |
| cancel_order | 885 | 85.9% |
| loans_and_credit | 1,050 | 90.0% |
| balance_or_statement | 1,074 | 90.8% |
| transfer_issue | 945 | 90.8% |
| change_delivery_details | 932 | 91.0% |
| unrecognized_transaction | 944 | 91.3% |
| card_not_working | 975 | 93.8% |
| fees_and_charges | 1,076 | 93.9% |
| handoff_to_human | 917 | 95.2% |
| sim_or_number | 1,005 | 95.5% |
| plan_change | 906 | 96.2% |
| network_or_internet_issue | 903 | 98.8% |
| account_access | 898 | 99.3% |

Most frequent disagreements (intended → judge):

- failed_delivery → order_status: 231
- other → handoff_to_human: 227
- order_status → failed_delivery: 148
- lost_or_stolen_card → card_not_working: 98
- missing_or_wrong_item → order_status: 86
- cancel_order → order_status: 78
- bill_inquiry → fees_and_charges: 70
- refund_request → failed_delivery: 64
- other → account_access: 59
- missing_or_wrong_item → other: 46

### Final SFT set

18,547 examples (train 17,579, val 968), 1,118,039 tokens (message + JSON target, danalm-v1).

| Variety | Examples | Avg words per message | Distinct word bigrams (messages) |
|---|---:|---:|---:|
| gulf_arabic | 5,152 | 9.3 | 0.52 |
| english | 5,614 | 10.8 | 0.42 |
| arabizi | 4,683 | 7.8 | 0.60 |
| mixed | 2,884 | 9.8 | 0.55 |
| saudi_arabic | 214 | 5.3 | 0.94 |

Examples per intent: min 760 (cancel_order), max 1,010 (fees_and_charges). Smallest intent × variety cells: failed_delivery/arabizi 46, other/mixed 60, plan_change/mixed 68, unrecognized_transaction/mixed 74, handoff_to_human/mixed 81.

Random training examples (seed 7):

| Variety | Intent | Message | Reply |
|---|---|---|---|
| gulf_arabic | roaming | عندي رومنج في مصر شو ليش ما اشوف النت؟ | سيتم تحويل رسالتك للفريق التقني لفحص حالة الاشتراك. |
| gulf_arabic | fees_and_charges | ايش السبب في fee الي اخذوا مني اليوم؟ ابا اعرف شو حيلته. | نقدر فهم استفسارك، نرجو الاتصال بنا لرفع الطلب للفريق المختص للتحقق. |
| gulf_arabic | missing_or_wrong_item | يا هلا، الطلب وصل بارد ومب زين، تبغى نرجعه الحين | نأسف للتأثير على وجبتكم. سنعرض ملاحظتكم على المختصين لمتابعة الحل معكم. |
| gulf_arabic | lost_or_stolen_card | شنو الوان الحين ما قدرت اجد الكرت عقب ما اخريته | نوصيك بطلب استبدال عبر التطبيق وسيتم تحويل الطلب للفريق المختص للتحقق من الوضع. |
| english | balance_or_statement | hello, i cant find my statement link on website, is it down? | I cannot check system status or view your account. Please try again later or contact our team for technical support. |
| english | failed_delivery | Hi, my order says delivered but I got nothing pls check asap | We are sorry to hear this. Please contact our support team so they can investigate for you. |
| english | balance_or_statement | Hi, where can I find my recent transactions in AED? | You can view all recent transactions by logging into your account. For more help, please reach out to our support desk. |
| english | roaming | Accidentally used data in Oman, huge bill coming! How stop this? | Please check your account options immediately to manage roaming usage or block it. |
| arabizi | balance_or_statement | abi arsal li tariq al7awajat last 3ashar ayam, laish ma yimken? | سأمرر رسالتكم للفريق المعني لمتابعة الأمر. يرجى الاتصال بخدمة العملاء لاستلام تفاصيل آخر عشر أيام. |
| arabizi | handoff_to_human | shlon t9dr? abi m3a shakhsi 2ashwan laish mob wayed al7een! | أفهم انزعاجك، سأقوم بربطك فورا بمسؤول لسماع ملاحظاتك. |
| arabizi | bill_inquiry | shlon al7een al3adad mobi? abgha a3ref laish wayed. | نأسف لوجود هذا الاستفسار، سنعرض ملاحظاتكم للفريق الفني للتحقق من الفاتورة. يرجى مراجعة قنوات الدعم الرسمية للحصول على توضيح إضافي. |
| arabizi | lost_or_stolen_card | Sara liya shi 7alwa b2al 5amla karti wabi3tha t7bati! | لا داعي للقلق، سنقوم بتمرير شكواك للفريق المختص فورا. الرجاء التواصل مع المركز لاستكمال الإجراءات الأمنية اللازمة. |
| mixed | balance_or_statement | ال transaction رقم 12345 ما يظهر في التقرير. | سأمرر هذا الرقم للفريق للتحقق منه فورا. |
| mixed | missing_or_wrong_item | طلب ما وصل وال app says delivered، فيه item مفقود. | سأقوم بمرور الطلب للفريق للتحقق منه. يرجى الانتظار قليلا. |
| mixed | missing_or_wrong_item | ال Item اللي طلبته ما في ال Box، ولا حتى ال Drink. | نعتذر، سيتم تحويل الأمر للفريق للتحقق من حالة الطلب. |
| mixed | balance_or_statement | أنا في دبي وأبغى check my account balance قبل الإجازة. | ال balance يظهر فورا عند تسجيل الدخول للتطبيق. إذا لم تظهر البيانات، سنعرض الأمر لفريق التقنية. |
| saudi_arabic | other | ما لا يعجبني | أعتذر عن عدم رضاك، سأمرر ملاحظاتك للفريق المختص لتحسين الخدمة. هل هناك شيء محدد تود التحدث عنه؟ |
| saudi_arabic | other | ضم هذا | وعليكم السلام، أهلا بكم. كيف يمكنني مساعدتكم اليوم؟ |
| saudi_arabic | other | حط منبه على سبعة الصبح | أهلا بك، نعتذر أننا لا نستطيع تنفيذ هذا الأمر مباشرة، وسنحوله للفريق المختص. |
| saudi_arabic | other | ودي قهوة الحين | أهلا وسهلا، طلبك بخصوص القهوة سيتم إحالته للفريق المختص لخدمتك. |
