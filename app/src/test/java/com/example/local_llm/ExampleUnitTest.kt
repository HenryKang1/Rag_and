package com.example.local_llm

import org.junit.Test
import org.junit.Assert.*

/**
 * Unit tests for local_llm application
 */
class LocalLLMUnitTest {
    @Test
    fun basicMathTest() {
        assertEquals(4, 2 + 2)
    }
    
    @Test
    fun stringValidationTest() {
        val testString = "Hello World"
        assertTrue("String should not be empty", testString.isNotEmpty())
        assertEquals("Hello World", testString)
    }
}