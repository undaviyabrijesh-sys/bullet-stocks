"""Small, explicit translations for StockMind's fixed Telegram interface.

Model-generated research remains in the language of the user's prompt. These
strings cover bot-owned UX only; explicit language preferences are persisted
locally, while conversation and beta state remain process-local.
"""

SUPPORTED_LANGUAGES = ("en", "hi", "gu")
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "हिन्दी",
    "gu": "ગુજરાતી",
}

_ALIASES = {
    "english": "en",
    "hindi": "hi",
    "हिंदी": "hi",
    "हिन्दी": "hi",
    "gujarati": "gu",
    "ગુજરાતી": "gu",
}


def normalize_language(value):
    """Normalize a supported language name or Telegram locale to its base code."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().replace("_", "-").casefold()
    if not normalized:
        return None
    if normalized in _ALIASES:
        return _ALIASES[normalized]
    base = normalized.split("-", 1)[0]
    return base if base in SUPPORTED_LANGUAGES else None


def resolve_language(explicit_preference=None, telegram_language_code=None):
    """Prefer an explicit user choice, then Telegram's locale, then English."""
    return (
        normalize_language(explicit_preference)
        or normalize_language(telegram_language_code)
        or "en"
    )


_MESSAGES = {
    "en": {
        "language_help": (
            "Choose StockMind's fixed interface language with /language en, "
            "/language hi or /language gu. Current language: {language_name}. "
            "Research answers still follow the language of your question."
        ),
        "language_invalid": "Supported languages: en (English), hi (Hindi), gu (Gujarati).",
        "language_changed": (
            "Interface language set to {language_name}. Research answers will still "
            "follow the language of your question."
        ),
        "language_changed_not_saved": (
            "Interface language set to {language_name} for this session, but could not be saved. "
            "It may reset when the bot restarts."
        ),
        "welcome": (
            "👋 Welcome to StockMind AI Beta!\n\n"
            "Ask about Indian stocks and crypto in English, Hinglish, Hindi, Gujarati or mixed language.\n\n"
            "You can also upload a chart screenshot for Demand & Supply (DNS), fresh/mitigated zones, "
            "BOS/CHOCH and liquidity analysis. When a zone is clear, I return an annotated copy with "
            "red supply and green demand bands.\n\n"
            "Examples:\n"
            "• Analyse Reliance fundamentals\n"
            "• Bhai HDFC Bank ka kya scene hai?\n"
            "• BTC 1H chart par demand supply draw karo\n"
            "• Is this demand zone fresh or mitigated?\n"
            "• Compare TCS and Infosys\n\n"
            "For chart screenshots, keep the price scale and timeframe visible. Add symbol, exchange "
            "and timeframe in your caption when they are not shown. Visual price levels may be estimates.\n\n"
            "I provide research, not trade execution or guaranteed buy/sell calls.\n"
            "Beta limit: 10 research questions per day."
        ),
        "beta_full": "The 10-person beta is currently full. Please try again later.",
        "daily_limit": "You've reached today's beta research limit. It resets at midnight IST.",
        "progress_chart": "🖼️ Reading the chart and mapping visible structure…",
        "progress_research": "🔎 Understanding your question and checking market data…",
        "chart_analysis_fallback": "I could not read the chart reliably.",
        "overlay_plot_missing": (
            "\n\nI could not locate the main price-plot boundary confidently, so I withheld all overlays. "
            "Keep the full candle panel and its edges visible in the screenshot."
        ),
        "overlay_no_zone": (
            "\n\nNo distinct supply/demand zone was returned from the visible chart. This does not prove "
            "that no zone exists; cropped history or unclear candles may hide it."
        ),
        "overlay_invalid_zone_objects": (
            "\n\nThe chart response contained no valid zone objects, so no overlay was drawn. "
            "Verify the written analysis against the screenshot."
        ),
        "overlay_not_drawable": (
            "\n\nThe analysis lists candidate areas, but none had geometry approved for drawing. "
            "Verify the written descriptions against the original screenshot; no overlay was guessed."
        ),
        "overlay_renderer_rejected": (
            "\n\nThe model supplied zone coordinates, but none passed the final pixel-size and "
            "plot-boundary checks. No overlay was drawn; use a sharper, uncropped screenshot and "
            "verify the written analysis."
        ),
        "overlay_render_failed": (
            "\n\nThe chart analysis completed, but I couldn't render the overlay. "
            "Please verify the described zones against the original screenshot."
        ),
        "chart_timeout": (
            "I stopped this chart analysis after its {seconds:g}-second time budget expired. "
            "No completed zone analysis or overlay was produced. Please retry; this request was "
            "not treated as a successful analysis."
        ),
        "research_timeout": (
            "I stopped this research turn after its {seconds:g}-second time budget expired. "
            "No completed research answer was produced. A provider request may already have been "
            "received; no further model or market-data calls were started. Please retry with a narrower question."
        ),
        "research_failed": (
            "I couldn't complete this analysis right now. For screenshots, check that "
            "OPENAI_VISION_MODEL is set to a vision-capable model and try again. "
            "No missing price level or market figure has been guessed."
        ),
        "overlay_caption": (
            "StockMind AI visual zone overlay — estimated from this screenshot; verify against the chart price scale."
        ),
        "reset_done": "Conversation context cleared. Ask a new research question.",
        "upload_too_large": "That image is too large. Please send an image under 5 MB.",
        "upload_download_failed": "I couldn't download that image. Please send it again.",
        "unsupported_document": (
            "Please attach a chart as a photo or as a JPEG, PNG or WEBP image file."
        ),
        "unsupported_message": (
            "Please send a text question or upload a chart screenshot as a photo/image. "
            "Voice support can be added separately."
        ),
        "chart_caption_default": (
            "Please analyse this chart screenshot for Demand and Supply zones, structure and "
            "conditional setups. Tell me if the symbol or timeframe is not visible."
        ),
    },
    "hi": {
        "language_help": (
            "StockMind की इंटरफ़ेस भाषा चुनने के लिए /language en, /language hi या "
            "/language gu भेजें। अभी की भाषा: {language_name}। रिसर्च का जवाब आपके सवाल की भाषा में रहेगा।"
        ),
        "language_invalid": "उपलब्ध भाषाएँ: en (English), hi (हिन्दी), gu (ગુજરાતી)।",
        "language_changed": (
            "इंटरफ़ेस की भाषा {language_name} कर दी गई है। रिसर्च का जवाब आपके सवाल की भाषा में ही रहेगा।"
        ),
        "language_changed_not_saved": (
            "इंटरफ़ेस भाषा {language_name} इस सत्र के लिए सेट की गई है, लेकिन इसे सहेजा नहीं जा सका। "
            "बॉट रीस्टार्ट होने पर यह बदल सकती है।"
        ),
        "welcome": (
            "👋 StockMind AI बीटा में आपका स्वागत है!\n\n"
            "भारतीय शेयरों और क्रिप्टो के बारे में English, Hinglish, हिन्दी, ગુજરાતી या मिली-जुली भाषा में पूछें।\n\n"
            "आप Demand & Supply (DNS), fresh/mitigated zones, BOS/CHOCH और liquidity analysis के लिए "
            "चार्ट स्क्रीनशॉट भी भेज सकते हैं। ज़ोन स्पष्ट होने पर लाल supply और हरे demand bands वाला "
            "annotated चार्ट मिलेगा।\n\n"
            "उदाहरण:\n"
            "• Reliance के fundamentals analyse करो\n"
            "• HDFC Bank का क्या scene है?\n"
            "• BTC 1H chart पर demand supply draw करो\n"
            "• क्या यह demand zone fresh है या mitigated?\n"
            "• TCS और Infosys की तुलना करो\n\n"
            "चार्ट में price scale और timeframe साफ़ दिखने चाहिए। Symbol, exchange और timeframe न दिखें "
            "तो caption में लिखें। स्क्रीनशॉट से पढ़े गए price levels अनुमानित हो सकते हैं।\n\n"
            "यह रिसर्च के लिए है, ट्रेड execute करने या पक्के buy/sell calls के लिए नहीं।\n"
            "बीटा सीमा: रोज़ 10 रिसर्च सवाल।"
        ),
        "beta_full": "10 लोगों की बीटा सीमा पूरी हो गई है। कृपया बाद में फिर कोशिश करें।",
        "daily_limit": "आज की बीटा रिसर्च सीमा पूरी हो गई है। यह हर रात 12 बजे IST पर रीसेट होती है।",
        "progress_chart": "🖼️ चार्ट पढ़ रहा हूँ और दिखाई देने वाली संरचना देख रहा हूँ…",
        "progress_research": "🔎 आपका सवाल समझ रहा हूँ और मार्केट डेटा जाँच रहा हूँ…",
        "chart_analysis_fallback": "मैं चार्ट को भरोसेमंद तरीके से पढ़ नहीं पाया।",
        "overlay_plot_missing": (
            "\n\nमुख्य price-plot की सीमा भरोसे से पहचान नहीं पाया, इसलिए कोई overlay नहीं बनाया। "
            "कृपया पूरा candle panel और उसके किनारे स्क्रीनशॉट में रखें।"
        ),
        "overlay_no_zone": (
            "\n\nदिख रहे चार्ट से कोई स्पष्ट supply/demand zone नहीं मिला। इसका मतलब यह नहीं कि "
            "कोई zone मौजूद नहीं है; cropped history या अस्पष्ट candles के कारण वह छिपा हो सकता है।"
        ),
        "overlay_invalid_zone_objects": (
            "\n\nचार्ट जवाब में कोई मान्य zone object नहीं था, इसलिए overlay नहीं बनाया। "
            "लिखे गए analysis को मूल स्क्रीनशॉट से मिलाएँ।"
        ),
        "overlay_not_drawable": (
            "\n\nAnalysis में candidate areas हैं, लेकिन किसी की geometry drawing के लिए मंज़ूर नहीं हुई। "
            "मूल स्क्रीनशॉट से विवरण जाँचें; कोई zone अनुमान से नहीं बनाया गया।"
        ),
        "overlay_renderer_rejected": (
            "\n\nदिए गए zone coordinates अंतिम pixel-size और plot-boundary जाँच में पास नहीं हुए। "
            "Overlay नहीं बनाया गया; साफ़, बिना crop वाला स्क्रीनशॉट भेजें और analysis जाँचें।"
        ),
        "overlay_render_failed": (
            "\n\nचार्ट analysis पूरा हुआ, लेकिन overlay नहीं बन पाया। कृपया बताए गए zones को मूल स्क्रीनशॉट से जाँचें।"
        ),
        "chart_timeout": (
            "समय सीमा {seconds:g} सेकंड पूरी होने पर चार्ट analysis रोक दिया गया। "
            "कोई पूरा zone analysis या overlay नहीं बना। कृपया दोबारा कोशिश करें; इसे सफल analysis नहीं माना गया।"
        ),
        "research_timeout": (
            "समय सीमा {seconds:g} सेकंड पूरी होने पर रिसर्च रोक दी गई। कोई पूरा जवाब नहीं बना। "
            "Provider को अनुरोध मिल चुका हो सकता है; आगे कोई model या market-data call नहीं की गई। "
            "कृपया छोटा सवाल पूछकर दोबारा कोशिश करें।"
        ),
        "research_failed": (
            "अभी analysis पूरा नहीं हो पाया। स्क्रीनशॉट के लिए जाँचें कि OPENAI_VISION_MODEL में "
            "image-capable model सेट है। कोई price level या market figure अनुमान से नहीं जोड़ा गया।"
        ),
        "overlay_caption": "StockMind AI का visual zone overlay — इस स्क्रीनशॉट से अनुमानित; chart price scale से जाँचें।",
        "reset_done": "बातचीत का context साफ़ कर दिया गया है। नया रिसर्च सवाल पूछें।",
        "upload_too_large": "यह image बहुत बड़ी है। कृपया 5 MB से छोटी image भेजें।",
        "upload_download_failed": "यह image डाउनलोड नहीं हो पाई। कृपया फिर से भेजें।",
        "unsupported_document": "कृपया chart को photo या JPEG, PNG अथवा WEBP image file के रूप में भेजें।",
        "unsupported_message": "कृपया text सवाल भेजें या chart screenshot को photo/image के रूप में upload करें। Voice support अलग से जोड़ा जा सकता है।",
        "chart_caption_default": "कृपया इस chart screenshot में Demand और Supply zones, structure और conditional setups का analysis करें। Symbol या timeframe न दिखे तो बताएँ।",
    },
    "gu": {
        "language_help": (
            "StockMind ની interface ભાષા પસંદ કરવા /language en, /language hi અથવા "
            "/language gu મોકલો. હાલની ભાષા: {language_name}. Research નો જવાબ તમારા પ્રશ્નની ભાષામાં રહેશે."
        ),
        "language_invalid": "ઉપલબ્ધ ભાષાઓ: en (English), hi (Hindi), gu (ગુજરાતી).",
        "language_changed": (
            "Interface ભાષા {language_name} રાખી છે. Research નો જવાબ તમારા પ્રશ્નની ભાષામાં જ રહેશે."
        ),
        "language_changed_not_saved": (
            "Interface ભાષા {language_name} આ સત્ર માટે રાખી છે, પરંતુ સાચવી શકાઈ નથી. "
            "Bot ફરી શરૂ થાય ત્યારે ભાષા બદલાઈ શકે છે."
        ),
        "welcome": (
            "👋 StockMind AI બીટામાં આપનું સ્વાગત છે!\n\n"
            "ભારતીય શેરો અને crypto વિશે English, Hinglish, Hindi, ગુજરાતી અથવા મિશ્ર ભાષામાં પૂછો.\n\n"
            "Demand & Supply (DNS), fresh/mitigated zones, BOS/CHOCH અને liquidity analysis માટે chart screenshot "
            "પણ મોકલી શકો છો. Zone સ્પષ્ટ હશે ત્યારે લાલ supply અને લીલા demand bands સાથે annotated chart મળશે.\n\n"
            "ઉદાહરણ:\n"
            "• Reliance ના fundamentals analyse કરો\n"
            "• HDFC Bank નો શું scene છે?\n"
            "• BTC 1H chart પર demand supply draw કરો\n"
            "• આ demand zone fresh છે કે mitigated?\n"
            "• TCS અને Infosys ની સરખામણી કરો\n\n"
            "Chart માં price scale અને timeframe સ્પષ્ટ રાખો. Symbol, exchange અથવા timeframe દેખાતા ન હોય "
            "તો caption માં લખો. Screenshot પરથી વાંચેલા price levels અંદાજિત હોઈ શકે છે.\n\n"
            "આ research માટે છે; trade execution અથવા ખાતરીવાળા buy/sell calls માટે નથી.\n"
            "બીટા મર્યાદા: દરરોજ 10 research પ્રશ્નો."
        ),
        "beta_full": "10 સભ્યોની beta મર્યાદા પૂર્ણ થઈ ગઈ છે. કૃપા કરીને પછી ફરી પ્રયાસ કરો.",
        "daily_limit": "આજની beta research મર્યાદા પૂર્ણ થઈ છે. તે દરરોજ રાત્રે 12 વાગ્યે IST પર ફરી શરૂ થાય છે.",
        "progress_chart": "🖼️ Chart વાંચી રહ્યો છું અને દેખાતી રચનાનું mapping કરું છું…",
        "progress_research": "🔎 તમારો પ્રશ્ન સમજી રહ્યો છું અને market data તપાસું છું…",
        "chart_analysis_fallback": "હું chart વિશ્વસનીય રીતે વાંચી શક્યો નથી.",
        "overlay_plot_missing": (
            "\n\nમુખ્ય price-plot ની સીમા વિશ્વાસપૂર્વક ઓળખી શક્યો નથી, તેથી overlay બનાવ્યું નથી. "
            "કૃપા કરીને આખું candle panel અને તેની કિનારીઓ screenshot માં દેખાય તે રીતે મોકલો."
        ),
        "overlay_no_zone": (
            "\n\nદેખાતા chart માંથી કોઈ સ્પષ્ટ supply/demand zone મળ્યો નથી. તેનો અર્થ zone નથી એવો નથી; "
            "cropped history અથવા અસ્પષ્ટ candles તેને છુપાવી શકે છે."
        ),
        "overlay_invalid_zone_objects": (
            "\n\nChart response માં માન્ય zone object નહોતો, તેથી overlay દોર્યું નથી. "
            "લખાયેલ analysis મૂળ screenshot સાથે તપાસો."
        ),
        "overlay_not_drawable": (
            "\n\nAnalysis માં candidate areas છે, પરંતુ કોઈની geometry drawing માટે મંજૂર થઈ નથી. "
            "મૂળ screenshot સામે વિગતો તપાસો; કોઈ zone અંદાજથી દોર્યો નથી."
        ),
        "overlay_renderer_rejected": (
            "\n\nઆપેલા zone coordinates અંતિમ pixel-size અને plot-boundary તપાસમાં પાસ થયા નથી. "
            "Overlay દોર્યું નથી; વધુ સ્પષ્ટ અને uncropped screenshot મોકલીને analysis તપાસો."
        ),
        "overlay_render_failed": (
            "\n\nChart analysis પૂર્ણ થયું, પરંતુ overlay બનાવી શક્યો નથી. કૃપા કરીને જણાવેલા zones મૂળ screenshot સામે તપાસો."
        ),
        "chart_timeout": (
            "{seconds:g} સેકન્ડની સમયમર્યાદા પૂરી થતાં chart analysis રોક્યું. "
            "પૂર્ણ zone analysis અથવા overlay બન્યું નથી. ફરી પ્રયાસ કરો; આને સફળ analysis ગણ્યું નથી."
        ),
        "research_timeout": (
            "{seconds:g} સેકન્ડની સમયમર્યાદા પૂરી થતાં research રોક્યું. પૂર્ણ જવાબ મળ્યો નથી. "
            "Provider ને વિનંતી મળી હોઈ શકે છે; ત્યારબાદ કોઈ model અથવા market-data call શરૂ કરી નથી. "
            "કૃપા કરીને વધુ ચોક્કસ અને નાનો પ્રશ્ન પૂછો."
        ),
        "research_failed": (
            "હાલ analysis પૂર્ણ કરી શક્યો નથી. Screenshot માટે OPENAI_VISION_MODEL માં image-capable model "
            "સેટ છે તેની તપાસો. કોઈ price level અથવા market figure અંદાજથી ઉમેર્યો નથી."
        ),
        "overlay_caption": "StockMind AI visual zone overlay — આ screenshot પરથી અંદાજિત; chart price scale સામે તપાસો.",
        "reset_done": "વાતચીતનો context સાફ કર્યો છે. નવો research પ્રશ્ન પૂછો.",
        "upload_too_large": "આ image ખૂબ મોટી છે. કૃપા કરીને 5 MB કરતાં નાની image મોકલો.",
        "upload_download_failed": "આ image download થઈ શકી નથી. કૃપા કરીને ફરી મોકલો.",
        "unsupported_document": "કૃપા કરીને chart ને photo અથવા JPEG, PNG કે WEBP image file તરીકે મોકલો.",
        "unsupported_message": "કૃપા કરીને text પ્રશ્ન મોકલો અથવા chart screenshot ને photo/image તરીકે upload કરો. Voice support અલગથી ઉમેરી શકાય છે.",
        "chart_caption_default": "કૃપા કરીને આ chart screenshot માં Demand અને Supply zones, structure અને conditional setups નું analysis કરો. Symbol અથવા timeframe ન દેખાય તો જણાવો.",
    },
}


def translate(language, key, **values):
    """Return a fixed UI string, falling back safely to English for unknown locales."""
    normalized = normalize_language(language) or "en"
    template = _MESSAGES[normalized].get(key)
    if template is None:
        template = _MESSAGES["en"].get(key)
    if template is None:
        raise KeyError(f"Unknown StockMind UI message: {key}")
    return template.format(**values)
