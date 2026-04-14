#include <jni.h>
#include <android/log.h>
#include <string>
#include <sstream>
#include "bench_session.h"
#include "MNN/expr/ExecutorScope.hpp"

#define LOG_TAG "MNNBench"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Helper: create a Java Long object
static jobject newJavaLong(JNIEnv* env, int64_t value) {
    jclass cls = env->FindClass("java/lang/Long");
    jmethodID init = env->GetMethodID(cls, "<init>", "(J)V");
    jobject obj = env->NewObject(cls, init, value);
    env->DeleteLocalRef(cls);
    return obj;
}

// Helper: build a HashMap with timing metrics from LlmContext
static jobject buildMetricsMap(JNIEnv* env, const MNN::Transformer::LlmContext* ctx) {
    jclass mapCls = env->FindClass("java/util/HashMap");
    jmethodID mapInit = env->GetMethodID(mapCls, "<init>", "()V");
    jmethodID putMethod = env->GetMethodID(mapCls, "put",
        "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jobject map = env->NewObject(mapCls, mapInit);

    auto put = [&](const char* key, int64_t val) {
        jstring jkey = env->NewStringUTF(key);
        jobject jval = newJavaLong(env, val);
        env->CallObjectMethod(map, putMethod, jkey, jval);
        env->DeleteLocalRef(jkey);
        env->DeleteLocalRef(jval);
    };

    if (ctx) {
        put("prompt_len", ctx->prompt_len);
        put("decode_len", ctx->gen_seq_len);
        put("vision_time", ctx->vision_us);
        put("prefill_time", ctx->prefill_us);
        put("decode_time", ctx->decode_us);
    }

    env->DeleteLocalRef(mapCls);
    return map;
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_mnn_benchmarkapp_jni_NativeBridge_initNative(
        JNIEnv* env, jobject /* thiz */,
        jstring modelDir, jstring configJson) {
    const char* dir = env->GetStringUTFChars(modelDir, nullptr);
    const char* cfg = env->GetStringUTFChars(configJson, nullptr);

    auto* session = new bench::BenchSession(dir, cfg);

    env->ReleaseStringUTFChars(modelDir, dir);
    env->ReleaseStringUTFChars(configJson, cfg);

    LOGD("Loading model from: %s", dir);
    bool ok = session->load();
    if (!ok || !session->isReady()) {
        LOGE("Model load failed");
        delete session;
        jclass exCls = env->FindClass("java/lang/IllegalStateException");
        if (exCls) env->ThrowNew(exCls, "Model load failed");
        return 0;
    }

    LOGD("Model loaded successfully, session=%p", session);
    return reinterpret_cast<jlong>(session);
}

JNIEXPORT jobject JNICALL
Java_com_mnn_benchmarkapp_jni_NativeBridge_generateNative(
        JNIEnv* env, jobject /* thiz */,
        jlong sessionPtr, jstring prompt, jobject progressListener) {
    auto* session = reinterpret_cast<bench::BenchSession*>(sessionPtr);
    if (!session || !session->isReady()) {
        return buildMetricsMap(env, nullptr);
    }

    const char* promptStr = env->GetStringUTFChars(prompt, nullptr);
    std::string promptCpp(promptStr);
    env->ReleaseStringUTFChars(prompt, promptStr);

    // Get callback method
    jclass listenerCls = env->GetObjectClass(progressListener);
    jmethodID onProgress = env->GetMethodID(listenerCls, "onProgress", "(Ljava/lang/String;)Z");

    auto* ctx = session->generate(promptCpp,
        [&](const std::string& token, bool is_eop) -> bool {
            if (progressListener && onProgress) {
                jstring jtoken = is_eop ? nullptr : env->NewStringUTF(token.c_str());
                jboolean cont = env->CallBooleanMethod(progressListener, onProgress, jtoken);
                if (jtoken) env->DeleteLocalRef(jtoken);
                return (bool)cont;
            }
            return true;
        });

    return buildMetricsMap(env, ctx);
}

JNIEXPORT jobject JNICALL
Java_com_mnn_benchmarkapp_jni_NativeBridge_generateWithImageNative(
        JNIEnv* env, jobject /* thiz */,
        jlong sessionPtr, jstring prompt, jstring imagePath, jobject progressListener) {
    auto* session = reinterpret_cast<bench::BenchSession*>(sessionPtr);
    if (!session || !session->isReady()) {
        return buildMetricsMap(env, nullptr);
    }

    const char* promptStr = env->GetStringUTFChars(prompt, nullptr);
    const char* imageStr = env->GetStringUTFChars(imagePath, nullptr);
    std::string promptCpp(promptStr);
    std::string imageCpp(imageStr);
    env->ReleaseStringUTFChars(prompt, promptStr);
    env->ReleaseStringUTFChars(imagePath, imageStr);

    jclass listenerCls = env->GetObjectClass(progressListener);
    jmethodID onProgress = env->GetMethodID(listenerCls, "onProgress", "(Ljava/lang/String;)Z");

    auto* ctx = session->generateWithImage(promptCpp, imageCpp,
        [&](const std::string& token, bool is_eop) -> bool {
            if (progressListener && onProgress) {
                jstring jtoken = is_eop ? nullptr : env->NewStringUTF(token.c_str());
                jboolean cont = env->CallBooleanMethod(progressListener, onProgress, jtoken);
                if (jtoken) env->DeleteLocalRef(jtoken);
                return (bool)cont;
            }
            return true;
        });

    return buildMetricsMap(env, ctx);
}

JNIEXPORT void JNICALL
Java_com_mnn_benchmarkapp_jni_NativeBridge_releaseNative(
        JNIEnv* /* env */, jobject /* thiz */, jlong sessionPtr) {
    auto* session = reinterpret_cast<bench::BenchSession*>(sessionPtr);
    if (session) {
        LOGD("Releasing session %p", session);
        delete session;
    }
}

JNIEXPORT void JNICALL
Java_com_mnn_benchmarkapp_jni_NativeBridge_resetNative(
        JNIEnv* /* env */, jobject /* thiz */, jlong sessionPtr) {
    auto* session = reinterpret_cast<bench::BenchSession*>(sessionPtr);
    if (session) {
        session->reset();
    }
}

JNIEXPORT void JNICALL
Java_com_mnn_benchmarkapp_jni_NativeBridge_setSystemPromptNative(
        JNIEnv* env, jobject /* thiz */,
        jlong sessionPtr, jstring prompt) {
    auto* session = reinterpret_cast<bench::BenchSession*>(sessionPtr);
    if (session && prompt) {
        const char* promptStr = env->GetStringUTFChars(prompt, nullptr);
        session->setSystemPrompt(std::string(promptStr));
        env->ReleaseStringUTFChars(prompt, promptStr);
    }
}

} // extern "C"
