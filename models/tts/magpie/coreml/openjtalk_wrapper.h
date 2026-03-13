/**
 * OpenJTalk G2P Wrapper
 *
 * Thin C API over OpenJTalk's frontend pipeline for Japanese text-to-phoneme conversion.
 * Returns NJD (Nihongo Jisho Data) features as JSON for pitch accent calculation in Swift.
 */

#ifndef OPENJTALK_WRAPPER_H
#define OPENJTALK_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize OpenJTalk with a MeCab dictionary.
 *
 * @param dict_path Path to open_jtalk_dic_utf_8-1.11 directory
 * @return 1 on success, 0 on failure
 */
int openjtalk_init(const char* dict_path);

/**
 * Run G2P frontend on Japanese text.
 *
 * Returns NJD word features as a JSON array string. Each element:
 *   {"string":"surface", "pos":"品詞", "pron":"カタカナ",
 *    "acc":N, "mora_size":N, "chain_flag":N}
 *
 * @param text UTF-8 encoded Japanese text
 * @return Newly allocated JSON string. Caller must free with openjtalk_free_string().
 *         Returns NULL on error.
 */
char* openjtalk_g2p(const char* text);

/**
 * Free a string returned by openjtalk_g2p().
 *
 * @param s String to free. Safe to call with NULL.
 */
void openjtalk_free_string(char* s);

/**
 * Release all OpenJTalk resources.
 * After calling this, openjtalk_init() must be called again before openjtalk_g2p().
 */
void openjtalk_destroy(void);

#ifdef __cplusplus
}
#endif

#endif /* OPENJTALK_WRAPPER_H */
