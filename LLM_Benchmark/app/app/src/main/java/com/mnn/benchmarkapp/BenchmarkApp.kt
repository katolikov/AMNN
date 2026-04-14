package com.mnn.benchmarkapp

import android.app.Application

class BenchmarkApp : Application() {
    override fun onCreate() {
        super.onCreate()
        instance = this
    }

    companion object {
        lateinit var instance: BenchmarkApp
            private set
    }
}
