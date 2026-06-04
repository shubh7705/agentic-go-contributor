// Copyright 2014 Manu Martinez-Almeida. All rights reserved.
// Use of this source code is governed by a MIT style
// license that can be found in the LICENSE file.

package gin

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"maps"
	"math"
	"mime/multipart"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/gin-contrib/sse"
	"github.com/gin-gonic/gin/binding"
	"github.com/gin-gonic/gin/render"
)

const defaultMultipartMemory = 32 << 20 // 32 MB

// Context is the most important part of gin. It allows us to pass variables between middleware,
// share parameters, route to specific handlers, validate the request, and render a response.
type Context struct {
	writermem responseWriter
	Request   *http.Request
	Writer    ResponseWriter

	Params   Params
	handlers HandlersChain
	index    int8
	fullPath string

	engine       *Engine
	params       *Params
	skippedNodes *[]skippedNode

	mu sync.RWMutex

	Keys map[any]any

	// Errors is a list of errors attached to all the handlers/middlewares that are used in the request.
	Errors errorMsgs

	accepted []string

	queryCache url.Values
	formCache  url.Values

	sameSite http.SameSite
}

func (c *Context) reset() {
	c.Writer = &c.writermem
	c.Params = c.Params[:0]
	c.handlers = nil
	c.index = -1
	c.fullPath = ""
	c.Keys = nil
	c.Errors = c.Errors[:0]
	c.accepted = nil
	c.queryCache = nil
	c.formCache = nil
	c.sameSite = 0
	c.params = nil
	c.skippedNodes = nil
}

func (c *Context) Copy() *Context {
	cp := Context{
		writermem: c.writermem,
		Request:   c.Request,
		Params:    c.Params,
		engine:    c.engine,
		Keys:      make(map[any]any, len(c.Keys)),
		Errors:    c.Errors,
		accepted:  c.accepted,
		queryCache: c.queryCache,
		formCache:  c.formCache,
		sameSite:   c.sameSite,
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	maps.Copy(cp.Keys, c.Keys)
	return &cp
}

func (c *Context) HandlerName() string {
	return nameOfFunction(c.handlers.Last())
}

func (c *Context) HandlerNames() []string {
	hn := make([]string, 0, len(c.handlers))
	for _, val := range c.handlers {
		hn = append(hn, nameOfFunction(val))
	}
	return hn
}

func (c *Context) Handler() HandlerFunc {
	return c.handlers.Last()
}

func (c *Context) FullPath() string {
	return c.fullPath
}

func (c *Context) SetFullPath(fullPath string) {
	c.fullPath = fullPath
}

func (c *Context) Next() {
	c.index++
	for c.index < int8(len(c.handlers)) {
		c.handlers[c.index](c)
		c.index++
	}
}

func (c *Context) IsAborted() bool {
	return c.index >= abortIndex
}

func (c *Context) Abort() {
	c.index = abortIndex
}

func (c *Context) AbortWithStatus(code int) {
	c.writermem.status = code
	c.Writer.WriteHeaderNow()
	c.Abort()
}

func (c *Context) AbortWithStatusJSON(code int, jsonObj any) {
	c.AbortWithStatus(code)
	c.JSON(-1, jsonObj)
}

func (c *Context) AbortWithError(code int, err error) *Error {
	c.AbortWithStatus(code)
	return c.Error(err)
}

// Error attaches an error to the current context. The error is pushed to a list of errors.
// It's a good idea to call Error for each error that occurred during the resolution of a request.
// A middleware can be used to collect all the errors and push them to a database together,
// print a log, or append it in the HTTP response.
// Error will panic if err is nil.
func (c *Context) Error(err error) *Error {
	if err == nil {
		panic("err is nil")
	}

	var parsedError *Error
	ok := errors.As(err, &parsedError)
	if !ok {
		parsedError = &Error{
			Err:  err,
			Type: ErrorTypePrivate,
		}
	}

	c.Errors = append(c.Errors, parsedError)
	return parsedError
}

/************************************/
/******** METADATA MANAGEMENT********/
/************************************/

// Set is used to store a new key/value pair exclusively for this context.
// It also lazy initializes c.Keys if it was not used previously.
func (c *Context) Set(key any, value any) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.Keys == nil {
		c.Keys = make(map[any]any)
	}

	c.Keys[key] = value
}

// Get returns the value for the given key, ie: (value, true).
// If the value does not exist it returns (nil, false)
func (c *Context) Get(key any) (value any, exists bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	value, exists = c.Keys[key]
	return
}

// MustGet returns the value for the given key if it exists, otherwise it panics.
func (c *Context) MustGet(key any) any {
	if value, exists := c.Get(key); exists {
		return value
	}
	panic(fmt.Sprintf("key %v does not exist", key))
}

func getTyped[T any](c *Context, key any) (res T) {
	if val, ok := c.Get(key); ok && val != nil {
		res, _ = val.(T)
	}
	return
}

// GetString returns the value associated with the key as a string.
func (c *Context) GetString(key any) string {
	return getTyped[string](c, key)
}

// GetBool returns the value associated with the key as a boolean.
func (c *Context) GetBool(key any) bool {
	return getTyped[bool](c, key)
}

// GetInt returns the value associated with the key as an integer.
func (c *Context) GetInt(key any) int {
	return getTyped[int](c, key)
}

// GetInt8 returns the value associated with the key as an integer 8.
func (c *Context) GetInt8(key any) int8 {
	return getTyped[int8](c, key)
}

// GetInt16 returns the value associated with the key as an integer 16.
func (c *Context) GetInt16(key any) int16 {
	return getTyped[int16](c, key)
}

// GetInt32 returns the value associated with the key as an integer 32.
func (c *Context) GetInt32(key any) int32 {
	return getTyped[int32](c, key)
}

// GetInt64 returns the value associated with the key as an integer 64.
func (c *Context) GetInt64(key any) int64 {
	return getTyped[int64](c, key)
}

// GetUint returns the value associated with the key as an unsigned integer.
func (c *Context) GetUint(key any) uint {
	return getTyped[uint](c, key)
}

// GetUint8 returns the value associated with the key as an unsigned integer 8.
func (c *Context) GetUint8(key any) uint8 {
	return getTyped[uint8](c, key)
}

// GetUint16 returns the value associated with the key as an unsigned integer 16.
func (c *Context) GetUint16(key any) uint16 {
	return getTyped[uint16](c, key)
}

// GetUint32 returns the value associated with the key as an unsigned integer 32.
func (c *Context) GetUint32(key any) uint32 {
	return getTyped[uint32](c, key)
}

// GetUint64 returns the value associated with the key as an unsigned integer 64.
func (c *Context) GetUint64(key any) uint64 {
	return getTyped[uint64](c, key)
}

// GetFloat32 returns the value associated with the key as a float32.
func (c *Context) GetFloat32(key any) float32 {
	return getTyped[float32](c, key)
}

// GetFloat64 returns the value associated with the key as a float64.
func (c *Context) GetFloat64(key any) float64 {
	return getTyped[float64](c, key)
}

// GetTime returns the value associated with the key as time.
func (c *Context) GetTime(key any) time.Time {
	return getTyped[time.Time](c, key)
}

// GetDuration returns the value associated with the key as a duration.
func (c *Context) GetDuration(key any) time.Duration {
	return getTyped[time.Duration](c, key)
}

// GetError returns the value associated with the key as an error.
func (c *Context) GetError(key any) error {
	return getTyped[error](c, key)
}

// GetIntSlice returns the value associated with the key as a slice of integers.
func (c *Context) GetIntSlice(key any) []int {
	return getTyped[[]int](c, key)
}

// GetStringSlice returns the value associated with the key as a slice of strings.
func (c *Context) GetStringSlice(key any) []string {
	return getTyped[[]string](c, key)
}

// GetStringMap returns the value associated with the key as a map of strings.
func (c *Context) GetStringMap(key any) map[string]any {
	return getTyped[map[string]any](c, key)
}

// GetStringMapString returns the value associated with the key as a map of strings.
func (c *Context) GetStringMapString(key any) map[string]string {
	return getTyped[map[string]string](c, key)
}

// GetStringMapStringSlice returns the value associated with the key as a map of string slices.
func (c *Context) GetStringMapStringSlice(key any) map[string][]string {
	return getTyped[map[string][]string](c, key)
}

/************************************/
/************ FILE UPLOAD ************/
/************************************/

// FormFile returns the first file for the provided form key.
func (c *Context) FormFile(name string) (*multipart.FileHeader, error) {
	if c.Request.MultipartForm == nil {
		if err := c.Request.ParseMultipartForm(c.engine.MaxMultipartMemory); err != nil {
			return nil, err
		}
	}
	f, fh, err := c.Request.FormFile(name)
	if err != nil {
		return nil, err
	}
	_ = f.Close()
	return fh, err
}

// SaveUploadedFile uploads the form file to specific dst.
func (c *Context) SaveUploadedFile(file *multipart.FileHeader, dst string) error {
	dir := filepath.Dir(dst)

	needChmod := true
	if _, err := os.Stat(dir); err == nil {
		needChmod = false
	} else if !os.IsNotExist(err) {
		return err
	}

	if err := os.MkdirAll(dir, c.engine.FileUploadDirPermission); err != nil {
		return err
	}

	if needChmod {
		if err := os.Chmod(dir, c.engine.FileUploadDirPermission); err != nil {
			return err
		}
	}

	src, err := file.Open()
	if err != nil {
		return err
	}
	defer src.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, src)
	return err
}
